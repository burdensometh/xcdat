"""Reconciliation of overlapping, explicitly-versioned multi-file datasets.

Data providers frequently publish a *new version* of a dataset that only
corrects a subset of an older observation period. When such files are read
together, the overlapping parts of the record must be resolved by an explicit
version (the highest version wins), while the parts of the older record that the
new file does not cover remain valid.

This module implements that reconciliation and keeps track of provenance:

* Overlapping segments are resolved by the **explicit version** attached to
  each file (a global attribute, an explicit mapping, or a callable). The
  version is never inferred from the input order or the file timestamps.
* Segments from the same version that overlap but disagree about a value are
  reported as **conflicts**. The involved files are identified and the
  conflicting interval is excluded from aggregation.
* Monthly statistics use the **actual duration** of every effective (valid)
  interval, so partial months are weighted by the number of seconds they cover.
* The result is independent of the input file order and is idempotent with
  respect to passing the same file more than once.

The reconciled :class:`xarray.Dataset` stores ``source_file`` and
``source_version`` as non-dimension coordinates along the time dimension.
Provenance and the conflict list are stored in the dataset attributes and can be
retrieved with :func:`version_provenance`, :func:`version_conflicts`, and
:func:`monthly_provenance`. Monthly means and valid durations are computed with
:func:`monthly_mean` and :func:`valid_duration`.
"""

import glob
import json
import os
import pathlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cf_xarray  # noqa: F401
import cftime
import numpy as np
import xarray as xr

from xcdat._logger import _setup_custom_logger
from xcdat.axis import get_dim_keys

logger = _setup_custom_logger(__name__)

#: The dataset attribute used to store reconciliation provenance and conflicts.
INFO_ATTR = "xcdat_version_info"
#: The reference epoch used to compare time values across the record.
TIME_UNITS = "microseconds since 1970-01-01 00:00:00"
#: Global attribute names inspected for an explicit version by default.
VERSION_ATTRS = ("version", "dataset_version", "revision")

#: Type alias for the accepted ``version`` specifications.
VersionInput = (
    None | str | Mapping[str, Any] | Callable[[str, xr.Dataset], Any] | Sequence[Any]
)


@dataclass(frozen=True)
class VersionSegment:
    """A valid interval of the reconciled record and its origin.

    Parameters
    ----------
    start : str
        The inclusive start of the interval as an ISO 8601 string.
    end : str
        The exclusive end of the interval as an ISO 8601 string.
    file : str
        The path of the file that supplies the interval.
    version : Any
        The explicit version of the source file.
    duration_seconds : float
        The length of the interval in seconds.
    """

    start: str
    end: str
    file: str
    version: Any
    duration_seconds: float


@dataclass(frozen=True)
class VersionConflict:
    """A same-version overlap whose values disagree.

    Parameters
    ----------
    start : str
        The inclusive start of the conflicting interval as an ISO 8601 string.
    end : str
        The exclusive end of the conflicting interval as an ISO 8601 string.
    version : Any
        The explicit version shared by the conflicting files.
    files : tuple[str, ...]
        The sorted paths of the files involved in the conflict.
    duration_seconds : float
        The length of the conflicting interval in seconds.
    """

    start: str
    end: str
    version: Any
    files: tuple[str, ...]
    duration_seconds: float


@dataclass(frozen=True)
class MonthlyProvenance:
    """The origin of the data that contributes to a calendar month.

    Parameters
    ----------
    month : str
        The calendar month (``"YYYY-MM"``).
    segments : tuple[VersionSegment, ...]
        The valid intervals that contribute to the month.
    valid_duration_seconds : float
        The total duration of valid data in the month, in seconds.
    conflict_duration_seconds : float
        The total duration excluded from the month due to conflicts, in seconds.
    """

    month: str
    segments: tuple[VersionSegment, ...]
    valid_duration_seconds: float
    conflict_duration_seconds: float


@dataclass
class _Interval:
    """A time interval supplied by a single source file."""

    start_us: int
    end_us: int
    version: Any
    version_key: tuple
    file: str
    interval_id: str
    values: xr.DataArray
    calendar: str | None


@dataclass
class _Piece:
    """An effective interval produced by resolving overlaps."""

    start_us: int
    end_us: int
    version: Any
    version_key: tuple
    file: str
    interval_id: str
    values: xr.DataArray


@dataclass
class _MonthTable:
    """Per-month bookkeeping for aggregation and provenance."""

    months: list[int] = field(default_factory=list)
    valid_seconds: dict[int, float] = field(default_factory=dict)
    conflict_seconds: dict[int, float] = field(default_factory=dict)
    calendar: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile_versions(
    paths: str | pathlib.Path | Sequence[str | pathlib.Path],
    data_var: str | None = None,
    *,
    version: VersionInput = None,
    version_attr: str = "version",
    decode_times: bool = True,
    add_bounds: Sequence[str] | None = ("X", "Y"),
    center_times: bool = False,
    lon_orient: tuple[float, float] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Reads and reconciles overlapping, explicitly-versioned files.

    Each file is assigned an explicit version. Where files overlap in time, the
    interval from the highest version wins; parts of the older record that are
    not covered by a newer file remain valid. Same-version overlaps with
    disagreeing values are recorded as conflicts and excluded from the result.

    Parameters
    ----------
    paths : str | pathlib.Path | Sequence[str | pathlib.Path]
        Paths to NetCDF files. Supported inputs match
        :py:func:`xcdat.open_mfdataset`: a directory (all ``*.nc`` files), a
        glob, a single path, or a list of paths.
    data_var : str | None, optional
        The key of the non-bounds data variable to reconcile. If ``None``, the
        data must contain exactly one data variable that depends on the time
        dimension.
    version : VersionInput, optional
        How the explicit version is obtained for each file. Options include:

          * ``None``: read the version from the global attribute named by
            ``version_attr`` (falling back to "dataset_version" and
            "revision").
          * ``str``: read the version from the global attribute of this name.
          * Mapping: map a path (absolute path, real path, or basename) to a
            version.
          * Callable: called as ``version(path, dataset)`` for each file.
          * Sequence: one version per input path, aligned with ``paths``.
    version_attr : str, optional
        The default global attribute to read the version from, by default
        "version".
    decode_times : bool, optional
        If True, decode time coordinates and bounds, by default True.
    add_bounds : Sequence[str] | None, optional
        CF axes for which missing bounds are added, by default ("X", "Y").
        Time bounds are derived from the time coordinates when absent.
    center_times : bool, optional
        If True, center time coordinates using their bounds, by default False.
    lon_orient : tuple[float, float] | None, optional
        The orientation to use for the longitude axis, by default None.
    **kwargs : Any
        Additional arguments passed to :py:func:`xcdat.open_dataset`.

    Returns
    -------
    xr.Dataset
        A reconciled dataset containing the data variable indexed by effective
        interval midpoints, ``time_bnds`` holding the interval bounds, the
        non-dimension coordinates ``source_file`` and ``source_version``, and
        the provenance/conflict information in its attributes.

    Raises
    ------
    ValueError
        If no input files are found, if the explicit version cannot be
        determined, or if the files use incompatible time or grid definitions.

    Examples
    --------
    Reconcile files that carry a ``"version"`` global attribute:

    >>> ds = xcdat.reconcile_versions("path/to/files", data_var="ts")
    >>> means = xcdat.monthly_mean(ds, "ts")
    """
    file_paths = _expand_paths(paths)
    if len(file_paths) == 0:
        raise ValueError(f"No NetCDF files were found for {paths!r}.")

    from xcdat.dataset import open_dataset

    count = len(file_paths)
    opened: list[tuple[str, Any, xr.Dataset]] = []
    for idx, path in enumerate(file_paths):
        ds = open_dataset(
            path,
            data_var=data_var,
            add_bounds=add_bounds,
            decode_times=decode_times,
            center_times=center_times,
            lon_orient=lon_orient,
            **kwargs,
        )
        resolved = _resolve_version(version, path, ds, idx, count, version_attr)
        opened.append((os.path.realpath(os.fspath(path)), resolved, ds))

    # De-duplicate identical files, keeping the highest explicit version. This
    # makes the operation idempotent with respect to duplicate inputs.
    selected: dict[str, tuple[Any, xr.Dataset]] = {}
    for real, resolved, ds in opened:
        if real not in selected or _version_key(resolved) > _version_key(
            selected[real][0]
        ):
            selected[real] = (resolved, ds)

    unique_paths = sorted(selected)
    datasets = [selected[path][1] for path in unique_paths]
    versions = [selected[path][0] for path in unique_paths]

    return reconcile_datasets(
        datasets, versions, data_var=data_var, source_names=unique_paths
    )


def reconcile_datasets(
    datasets: Sequence[xr.Dataset],
    versions: Sequence[Any],
    data_var: str | None = None,
    *,
    source_names: Sequence[str] | None = None,
) -> xr.Dataset:
    """Reconciles overlapping datasets that have explicit versions.

    Parameters
    ----------
    datasets : Sequence[xr.Dataset]
        The datasets to reconcile. Each dataset must contain the same data
        variable on a common time axis and, if applicable, a common grid.
    versions : Sequence[Any]
        One explicit version per dataset. Higher versions win where datasets
        overlap in time.
    data_var : str | None, optional
        The key of the non-bounds data variable to reconcile, by default None
        (inferred from the datasets).
    source_names : Sequence[str] | None, optional
        Names used for provenance instead of the datasets' ``source`` encoding.
        Must be the same length as ``datasets``.

    Returns
    -------
    xr.Dataset
        The reconciled dataset.

    Raises
    ------
    ValueError
        If the number of versions or source names does not match the number of
        datasets, or if the time/grid definitions are incompatible.
    """
    if len(datasets) != len(versions):
        raise ValueError(
            f"Received {len(datasets)} datasets but {len(versions)} versions."
        )
    if source_names is not None and len(source_names) != len(datasets):
        raise ValueError(
            f"Received {len(datasets)} datasets but {len(source_names)} source names."
        )
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required to reconcile.")

    names = (
        list(source_names)
        if source_names is not None
        else [
            str(ds.encoding.get("source", f"dataset_{i}"))
            for i, ds in enumerate(datasets)
        ]
    )
    data_var = _infer_data_var(datasets, data_var)

    intervals: list[_Interval] = []
    for ds, ver, name in zip(datasets, versions, names):
        intervals.extend(_intervals_from_dataset(ds, data_var, ver, name))

    calendar = _validate_calendar(intervals)
    _validate_grid(intervals)
    return _reconcile_intervals(intervals, data_var, calendar, datasets[0])


def monthly_mean(ds: xr.Dataset, data_var: str | None = None) -> xr.Dataset:
    """Computes duration-weighted monthly means and monthly valid durations.

    Each effective interval contributes to a month in proportion to the number
    of seconds it covers within that month. Conflicting intervals (recorded by
    :func:`reconcile_versions`) are absent from the dataset, so they do not
    contribute.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with a time dimension and time bounds. This is typically the
        output of :func:`reconcile_versions`, but any dataset with time bounds
        is accepted.
    data_var : str | None, optional
        The key of the data variable to average, by default None (inferred).

    Returns
    -------
    xr.Dataset
        A dataset indexed by the first day of each month containing the mean,
        ``valid_duration`` and ``conflict_duration`` (both in seconds).
    """
    layout = _time_layout(ds, data_var)
    table = _month_table(ds, layout)
    result = _monthly_means(ds, layout, table)
    return _assemble_monthly(ds, layout, table, result)


def valid_duration(ds: xr.Dataset, data_var: str | None = None) -> xr.DataArray:
    """Returns the valid (non-conflicting) duration of each calendar month.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with time bounds. Conflicting intervals are absent from a
        reconciled dataset, so they are excluded automatically.
    data_var : str | None, optional
        The key of the data variable used to disambiguate time bounds, by
        default None.

    Returns
    -------
    xr.DataArray
        A DataArray indexed by the first day of each month, in seconds, named
        "valid_duration".
    """
    layout = _time_layout(ds, data_var)
    table = _month_table(ds, layout)
    return _duration_array(layout, table, "valid_duration")


def version_provenance(ds: xr.Dataset) -> list[VersionSegment]:
    """Returns the origin of every valid interval in a reconciled dataset.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset produced by :func:`reconcile_versions`.

    Returns
    -------
    list[VersionSegment]
        The provenance of each valid interval, sorted by start time. An empty
        list is returned when the dataset does not carry reconciliation
        information.
    """
    return [
        _segment_from_item(item)
        for item in _read_info(ds).get("segments", [])
    ]


def version_conflicts(ds: xr.Dataset) -> list[VersionConflict]:
    """Returns the same-version value conflicts recorded for a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset produced by :func:`reconcile_versions`.

    Returns
    -------
    list[VersionConflict]
        The conflicts, sorted by start time. An empty list is returned when the
        dataset does not carry reconciliation information.
    """
    return [
        VersionConflict(
            start=str(item["start"]),
            end=str(item["end"]),
            version=item["version"],
            files=tuple(str(name) for name in item["files"]),
            duration_seconds=float(item["duration_seconds"]),
        )
        for item in _read_info(ds).get("conflicts", [])
    ]


def monthly_provenance(ds: xr.Dataset) -> list[MonthlyProvenance]:
    """Returns the origin of the data in every calendar month.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset produced by :func:`reconcile_versions`.

    Returns
    -------
    list[MonthlyProvenance]
        One entry per month that has valid data or a conflict, sorted by month.
    """
    info = _read_info(ds)
    calendar = info.get("calendar")
    intervals = _info_intervals(info)

    layout = _layout_from_intervals(intervals, calendar)
    table = _month_table(ds, layout, intervals=intervals)

    segments: dict[int, list[VersionSegment]] = {}
    for interval in intervals:
        start = int(interval["start_us"])
        end = int(interval["end_us"])
        month = _month_start_us(start, calendar)
        while month < end:
            month_end = _next_month_us(month, calendar)
            lo = max(start, month)
            hi = min(end, month_end)
            if hi > lo:
                segments.setdefault(month, []).append(
                    VersionSegment(
                        start=_iso(lo, calendar),
                        end=_iso(hi, calendar),
                        file=str(interval["file"]),
                        version=interval["version"],
                        duration_seconds=(hi - lo) / 1e6,
                    )
                )
            month = month_end

    results: list[MonthlyProvenance] = []
    for month in table.months:
        results.append(
            MonthlyProvenance(
                month=_month_label(month, calendar),
                segments=tuple(segments.get(month, ())),
                valid_duration_seconds=float(table.valid_seconds.get(month, 0.0)),
                conflict_duration_seconds=float(
                    table.conflict_seconds.get(month, 0.0)
                ),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Path and version handling
# ---------------------------------------------------------------------------


def _expand_paths(
    paths: str | pathlib.Path | Sequence[str | pathlib.Path],
) -> list[str]:
    """Expands paths using the same rules as ``open_mfdataset``.

    Directory and glob inputs are sorted for determinism. Explicit sequences
    preserve their input order so ``version=`` sequences stay aligned.
    """
    if isinstance(paths, (str, pathlib.Path)):
        if os.path.isdir(paths):
            return sorted(glob.glob(os.path.join(os.fspath(paths), "*.nc")))
        return sorted(glob.glob(os.fspath(paths)))

    expanded: list[str] = []
    for item in paths:
        expanded.extend(_expand_paths(item))
    return expanded


def _resolve_version(
    spec: VersionInput,
    path: str,
    ds: xr.Dataset,
    index: int,
    count: int,
    version_attr: str,
) -> Any:
    """Resolves the explicit version of a single file."""
    if spec is None or isinstance(spec, str):
        return _version_from_attrs(ds, spec or version_attr)
    if callable(spec):
        return spec(path, ds)
    if isinstance(spec, Mapping):
        return _version_from_mapping(spec, path)
    if isinstance(spec, Sequence):
        if len(spec) != count:
            raise ValueError(
                f"Received {count} files but {len(spec)} versions. Provide one "
                "version per file."
            )
        return spec[index]
    raise TypeError(f"Unsupported version specification: {type(spec)!r}.")


def _version_from_attrs(ds: xr.Dataset, attr: str) -> Any:
    """Reads a version from a global attribute."""
    candidates = [attr]
    candidates.extend(name for name in VERSION_ATTRS if name not in candidates)
    for name in candidates:
        if name in ds.attrs:
            return ds.attrs[name]
    raise ValueError(
        "Could not determine an explicit version for a file. Expected a "
        f"global attribute named {attr!r} (or one of {VERSION_ATTRS}). Pass "
        "`version=` explicitly to `reconcile_versions()`."
    )


def _version_from_mapping(mapping: Mapping[str, Any], path: str) -> Any:
    """Reads a version from an explicit path mapping."""
    real = os.path.realpath(path)
    for key in (path, real, os.path.basename(path), os.path.basename(real)):
        if key in mapping:
            return mapping[key]
    raise ValueError(
        f"No explicit version was provided for {path!r}. Add it to the "
        "`version` mapping."
    )


# ---------------------------------------------------------------------------
# Interval extraction
# ---------------------------------------------------------------------------


def _infer_data_var(datasets: Sequence[xr.Dataset], data_var: str | None) -> str:
    """Infers the single time-dependent data variable."""
    if data_var is not None:
        return data_var

    ds = datasets[0]
    time_name = _time_name(ds)
    bounds_keys = _bounds_keys(ds)
    candidates = [
        name
        for name in ds.data_vars
        if time_name in ds[name].dims and name not in bounds_keys
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Could not infer a single data variable to reconcile. Found "
            f"{candidates}. Pass `data_var=` explicitly."
        )
    return str(candidates[0])


def _time_name(ds: xr.Dataset) -> str:
    """Returns the name of the time dimension."""
    key = get_dim_keys(ds, "T")
    if isinstance(key, list):
        raise ValueError(f"Expected a single time dimension but found multiple: {key}.")
    return key


def _bounds_keys(ds: xr.Dataset) -> set[str]:
    """Returns the names of the variables used as coordinate bounds."""
    keys: set[str] = set()
    for name in ds.coords:
        bound = ds[name].attrs.get("bounds")
        if bound:
            keys.add(str(bound))
    try:
        for names in ds.cf.bounds.values():
            keys.update(str(name) for name in names)
    except (AttributeError, KeyError):
        pass
    return keys


def _intervals_from_dataset(
    ds: xr.Dataset, data_var: str, version: Any, source: str
) -> list[_Interval]:
    """Extracts versioned time intervals from a single dataset."""
    time_name = _time_name(ds)
    if data_var not in ds:
        raise ValueError(f"The data variable {data_var!r} is not in the dataset.")
    if time_name not in ds[data_var].dims:
        raise ValueError(
            f"The data variable {data_var!r} does not depend on the time "
            f"dimension {time_name!r}."
        )

    time_values = ds[time_name].values
    calendar = _calendar(time_values)
    starts, ends = _interval_bounds_us(ds, time_name, calendar)

    version_key = _version_key(version)
    intervals: list[_Interval] = []
    for idx in range(ds[time_name].size):
        intervals.append(
            _Interval(
                start_us=starts[idx],
                end_us=ends[idx],
                version=version,
                version_key=version_key,
                file=source,
                interval_id=f"{source}@{starts[idx]}-{ends[idx]}",
                values=ds[data_var].isel({time_name: idx}, drop=True),
                calendar=calendar,
            )
        )
    return intervals


def _interval_bounds_us(
    ds: xr.Dataset, time_name: str, calendar: str | None
) -> tuple[list[int], list[int]]:
    """Returns the start/end of every time interval in microseconds."""
    time = ds[time_name]
    bounds = _find_time_bounds(ds, time_name)

    if bounds is not None:
        raw = np.asarray(bounds.values)
        if len(raw) != time.size:
            raise ValueError(
                f"The time bounds for {time_name!r} have {len(raw)} rows but "
                f"the time axis has {time.size} values."
            )
        starts: list[int] = []
        ends: list[int] = []
        for row in raw:
            micros = [_to_us(value, calendar) for value in np.ravel(row)]
            starts.append(min(micros))
            ends.append(max(micros))
        return starts, ends

    logger.warning(
        f"Time bounds were not found for {time_name!r}; deriving them from the "
        "time coordinates."
    )
    return _derive_bounds_us(time.values, calendar)


def _find_time_bounds(ds: xr.Dataset, time_name: str) -> xr.DataArray | None:
    """Locates the time bounds variable for a time coordinate."""
    attr = ds[time_name].attrs.get("bounds")
    if attr and str(attr) in ds.variables:
        return ds[str(attr)]

    try:
        keys = ds.cf.bounds.get("T", [])
    except (AttributeError, KeyError):
        keys = []
    for key in keys:
        if key in ds.variables:
            return ds[key]

    return None


def _derive_bounds_us(
    values: np.ndarray, calendar: str | None
) -> tuple[list[int], list[int]]:
    """Derives interval bounds from time coordinates.

    Time coordinates that fall in distinct months at roughly monthly spacing are
    treated as monthly centroids, so their bounds cover the whole month. Other
    coordinates use midpoints between neighboring values.
    """
    points = [_to_us(value, calendar) for value in values]
    count = len(points)
    if count == 0:
        return [], []
    if count == 1:
        start = _month_start_us(points[0], calendar)
        return [start], [_next_month_us(start, calendar)]

    if _looks_monthly(points, calendar):
        starts = [_month_start_us(point, calendar) for point in points]
        ends = [_next_month_us(start, calendar) for start in starts]
        return starts, ends

    starts = [0] * count
    ends = [0] * count
    for idx in range(count - 1):
        boundary = (points[idx] + points[idx + 1]) // 2
        ends[idx] = boundary
        starts[idx + 1] = boundary
    starts[0] = points[0] - (points[1] - points[0]) // 2
    ends[-1] = points[-1] + (points[-1] - points[-2]) // 2
    return starts, ends


def _looks_monthly(points: list[int], calendar: str | None) -> bool:
    """Returns True if the points are one-per-month, at roughly monthly spacing."""
    months = [_month_start_us(point, calendar) for point in points]
    if len(set(months)) != len(points):
        return False
    step = 20 * 86400 * 1_000_000
    for left, right in zip(points, points[1:]):
        if right - left < step:
            return False
    return True


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _validate_calendar(intervals: list[_Interval]) -> str | None:
    """Validates that all datasets share one time type and calendar."""
    calendars = {interval.calendar for interval in intervals}
    if len(calendars) != 1:
        raise ValueError(
            "Cannot reconcile datasets with mixed time types or calendars: "
            f"{sorted(str(c) for c in calendars)}."
        )
    return calendars.pop()


def _validate_grid(intervals: list[_Interval]) -> None:
    """Validates that all intervals share an identical non-time grid."""
    reference = intervals[0].values
    reference_dims = list(reference.dims)
    for interval in intervals[1:]:
        if list(interval.values.dims) != reference_dims:
            raise ValueError(
                "Cannot reconcile datasets with different data dimensions: "
                f"{reference_dims} vs {list(interval.values.dims)}."
            )
        try:
            xr.align(reference, interval.values, join="exact", copy=False)
        except (ValueError, xr.AlignmentError) as err:
            raise ValueError(
                f"Cannot reconcile {interval.file!r} with {intervals[0].file!r}: "
                "the non-time coordinates differ."
            ) from err


def _reconcile_intervals(
    intervals: list[_Interval],
    data_var: str,
    calendar: str | None,
    template: xr.Dataset,
) -> xr.Dataset:
    """Resolves overlapping intervals and builds the reconciled dataset."""
    if len(intervals) == 0:
        raise ValueError("No time intervals were found in the input datasets.")

    intervals.sort(key=lambda iv: (iv.file, iv.start_us, iv.end_us))
    boundaries = _partition_boundaries(intervals)
    conflicts, pieces = _resolve_segments(intervals, boundaries)
    pieces = _merge_pieces(pieces)
    conflicts = _merge_conflicts(conflicts)

    return _build_dataset(pieces, conflicts, data_var, calendar, template)


def _partition_boundaries(intervals: list[_Interval]) -> list[int]:
    """Builds the ordered set of every interval start and end."""
    boundaries: set[int] = set()
    for interval in intervals:
        boundaries.add(interval.start_us)
        boundaries.add(interval.end_us)
    return sorted(boundaries)


def _resolve_segments(
    intervals: list[_Interval], boundaries: list[int]
) -> tuple[list[dict[str, Any]], list[_Piece]]:
    """Resolves the effective source of each elementary interval."""
    conflicts: list[dict[str, Any]] = []
    pieces: list[_Piece] = []

    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if end <= start:
            continue
        covering = [
            interval
            for interval in intervals
            if interval.start_us <= start and interval.end_us >= end
        ]
        if len(covering) == 0:
            continue

        version_key, tops = _top_version(covering)
        if _conflicting_values(tops):
            conflicts.append(
                {
                    "start_us": start,
                    "end_us": end,
                    "version": tops[0].version,
                    "files": sorted({interval.file for interval in tops}),
                }
            )
            continue

        merged = _merge_values([interval.values for interval in tops])
        chosen = _attribute_interval(tops, merged)
        pieces.append(
            _Piece(
                start_us=start,
                end_us=end,
                version=chosen.version,
                version_key=version_key,
                file=chosen.file,
                interval_id=chosen.interval_id,
                values=merged,
            )
        )
    return conflicts, pieces


def _top_version(covering: list[_Interval]) -> tuple[tuple, list[_Interval]]:
    """Returns the highest version key and the intervals that carry it."""
    top = max(interval.version_key for interval in covering)
    tops = [interval for interval in covering if interval.version_key == top]
    tops.sort(key=lambda iv: iv.file)
    return top, tops


def _conflicting_values(intervals: list[_Interval]) -> bool:
    """Returns True if any same-version pair disagrees about a value."""
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            if _values_conflict(intervals[i].values, intervals[j].values):
                return True
    return False


def _values_conflict(a: xr.DataArray, b: xr.DataArray) -> bool:
    """Returns True if two value arrays disagree where both are defined."""
    if list(a.dims) != list(b.dims):
        return True
    try:
        left, right = xr.align(a, b, join="exact", copy=False)
    except (ValueError, xr.AlignmentError):
        return True
    try:
        differ = left.notnull() & right.notnull() & (left != right)
        return bool(differ.any())
    except (TypeError, ValueError):  # pragma: no cover - non-comparable dtypes
        return True


def _merge_values(values: list[xr.DataArray]) -> xr.DataArray:
    """Combines non-null values from several same-version arrays."""
    merged = values[0]
    for other in values[1:]:
        try:
            left, right = xr.align(merged, other, join="exact", copy=False)
        except (ValueError, xr.AlignmentError):  # pragma: no cover - defensive
            continue
        merged = xr.where(left.notnull(), left, right)
    return merged


def _attribute_interval(intervals: list[_Interval], merged: xr.DataArray) -> _Interval:
    """Chooses a deterministic source interval for a merged interval."""
    for interval in intervals:
        try:
            if interval.values.equals(merged):
                return interval
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    return intervals[0]


def _merge_pieces(pieces: list[_Piece]) -> list[_Piece]:
    """Merges adjacent pieces that originate from the same source interval."""
    if len(pieces) == 0:
        return []
    pieces = sorted(pieces, key=lambda p: (p.start_us, p.file))
    merged: list[_Piece] = [pieces[0]]
    for piece in pieces[1:]:
        last = merged[-1]
        same_source = (
            piece.interval_id == last.interval_id
            and piece.file == last.file
            and piece.version == last.version
            and piece.start_us == last.end_us
        )
        if same_source:
            last.end_us = piece.end_us
        else:
            merged.append(piece)
    return merged


def _merge_conflicts(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merges adjacent conflicts that involve the same files and version."""
    if len(conflicts) == 0:
        return []
    conflicts = sorted(conflicts, key=lambda c: (c["start_us"], tuple(c["files"])))
    merged: list[dict[str, Any]] = [conflicts[0]]
    for conflict in conflicts[1:]:
        last = merged[-1]
        if (
            conflict["start_us"] == last["end_us"]
            and conflict["files"] == last["files"]
            and conflict["version"] == last["version"]
        ):
            last["end_us"] = conflict["end_us"]
        else:
            merged.append(conflict)
    return merged


def _build_dataset(
    pieces: list[_Piece],
    conflicts: list[dict[str, Any]],
    data_var: str,
    calendar: str | None,
    template: xr.Dataset,
) -> xr.Dataset:
    """Builds the reconciled dataset from resolved pieces and conflicts."""
    info: dict[str, Any] = {
        "data_var": data_var,
        "calendar": calendar,
        "units": TIME_UNITS,
        "segments": [_segment_item(piece, calendar) for piece in pieces],
        "conflicts": [_conflict_item(conflict, calendar) for conflict in conflicts],
    }

    time_name = _time_name(template)
    bounds_name = _time_bounds_name(template, time_name)
    # Inherit non-time variables (static fields, grid coordinates and bounds).
    time_dependent = [
        name for name in template.variables if time_name in template[name].dims
    ]
    base = template.drop_vars(time_dependent)

    if len(pieces) == 0:
        # Keep an empty, typed time axis so consumers (e.g. ``monthly_mean``)
        # can still interpret the record and report the conflicts.
        ds = base.copy()
        ds = ds.assign_coords(
            {
                time_name: (time_name, np.array([], dtype=object)),
                bounds_name: ((time_name, "bnds"), np.empty((0, 2), dtype=object)),
                "source_file": (time_name, np.array([], dtype=object)),
                "source_version": (time_name, np.array([], dtype=object)),
            }
        )
        ds[time_name].attrs.update(
            {
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": bounds_name,
            }
        )
        ds[bounds_name].attrs["xcdat_bounds"] = "True"
        ds.attrs[INFO_ATTR] = json.dumps(info, sort_keys=True)
        return ds

    expanded = []
    for piece in pieces:
        label = _from_us((piece.start_us + piece.end_us) // 2, calendar)
        expanded.append(piece.values.expand_dims({time_name: [label]}))
    values = xr.concat(expanded, dim=time_name)

    labels = np.array(
        [_from_us((p.start_us + p.end_us) // 2, calendar) for p in pieces],
        dtype=object,
    )
    bounds = np.array(
        [
            [_from_us(p.start_us, calendar), _from_us(p.end_us, calendar)]
            for p in pieces
        ],
        dtype=object,
    )

    ds = base.assign_coords(
        {
            time_name: (time_name, labels),
            bounds_name: ((time_name, "bnds"), bounds),
            "source_file": (
                time_name,
                np.array([p.file for p in pieces], dtype=object),
            ),
            "source_version": (
                time_name,
                np.array([_display_version(p.version) for p in pieces], dtype=object),
            ),
        }
    )
    ds[data_var] = values
    ds[time_name].attrs.update(
        {
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": bounds_name,
        }
    )
    ds[bounds_name].attrs["xcdat_bounds"] = "True"
    ds.attrs[INFO_ATTR] = json.dumps(info, sort_keys=True)
    return ds


def _time_bounds_name(ds: xr.Dataset, time_name: str) -> str:
    """Returns the name to use for the time bounds variable."""
    attr = ds[time_name].attrs.get("bounds")
    if attr:
        return str(attr)
    return f"{time_name}_bnds"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _calendar(values: Any) -> str | None:
    """Returns the calendar of a datetime array, if it is cftime based."""
    array = np.asarray(values)
    if array.size == 0:
        return None
    first = array.ravel()[0]
    if isinstance(first, cftime.datetime):
        return str(first.calendar)
    return None


def _to_us(value: Any, calendar: str | None) -> int:
    """Converts a datetime-like value to integer microseconds since 1970."""
    if isinstance(value, cftime.datetime):
        cal = calendar or str(value.calendar)
        return int(round(cftime.date2num(value, TIME_UNITS, calendar=cal)))
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[us]").astype("int64"))
    return int(np.datetime64(value, "us").astype("int64"))


def _from_us(value: int, calendar: str | None) -> Any:
    """Converts integer microseconds since 1970 to a datetime-like value."""
    if calendar is None:
        return np.datetime64(int(value), "us")
    return cftime.num2date(
        int(value), TIME_UNITS, calendar=calendar, only_use_cftime_datetimes=True
    )


def _month_start_us(value: int, calendar: str | None) -> int:
    """Returns the start of the month containing a time value."""
    if calendar is None:
        month = np.datetime64(int(value), "us").astype("datetime64[M]")
        return int(month.astype("datetime64[us]").astype("int64"))
    dt = _from_us(value, calendar)
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return _to_us(start, calendar)


def _next_month_us(value: int, calendar: str | None) -> int:
    """Returns the start of the month following a month start."""
    if calendar is None:
        month = np.datetime64(int(value), "us").astype("datetime64[M]")
        month = month + np.timedelta64(1, "M")
        return int(month.astype("datetime64[us]").astype("int64"))
    dt = _from_us(value, calendar)
    if dt.month == 12:
        following = dt.replace(year=dt.year + 1, month=1)
    else:
        following = dt.replace(month=dt.month + 1)
    return _to_us(following, calendar)


def _month_label(value: int, calendar: str | None) -> str:
    """Returns a ``YYYY-MM`` label for a month start."""
    if calendar is None:
        dt = np.datetime64(int(value), "us").astype("datetime64[M]")
        return str(dt)
    local = _from_us(value, calendar)
    return f"{local.year:04d}-{local.month:02d}"


def _iso(value: int, calendar: str | None) -> str:
    """Formats a time value as an ISO 8601 string."""
    if calendar is None:
        return np.datetime_as_string(np.datetime64(int(value), "us"))
    return _from_us(value, calendar).isoformat()


def _version_key(version: Any) -> tuple:
    """Converts a version into a deterministically orderable key."""
    if version is None:
        return (-1, ())
    if isinstance(version, (int, np.integer, float, np.floating, bool)):
        return (0, (float(version),))
    parts: list[tuple[int, Any]] = []
    for part in re.split(r"(\d+)", str(version).strip()):
        if part == "":
            continue
        parts.append((1, int(part)) if part.isdigit() else (0, part))
    return (1, tuple(parts))


def _jsonable_version(version: Any) -> Any:
    """Converts a version to a JSON-serializable value."""
    if isinstance(version, np.integer):
        return int(version)
    if isinstance(version, np.floating):
        return float(version)
    if isinstance(version, (str, int, float, bool)) or version is None:
        return version
    return str(version)


def _display_version(version: Any) -> str:
    """Returns the string stored in the ``source_version`` coordinate."""
    return "" if version is None else str(version)


def _segment_item(piece: _Piece, calendar: str | None) -> dict[str, Any]:
    """Serializes a piece for the provenance attribute."""
    return {
        "start": _iso(piece.start_us, calendar),
        "end": _iso(piece.end_us, calendar),
        "start_us": piece.start_us,
        "end_us": piece.end_us,
        "file": piece.file,
        "version": _jsonable_version(piece.version),
        "duration_seconds": (piece.end_us - piece.start_us) / 1e6,
    }


def _conflict_item(conflict: dict[str, Any], calendar: str | None) -> dict[str, Any]:
    """Serializes a conflict for the provenance attribute."""
    return {
        "start": _iso(conflict["start_us"], calendar),
        "end": _iso(conflict["end_us"], calendar),
        "start_us": conflict["start_us"],
        "end_us": conflict["end_us"],
        "version": _jsonable_version(conflict["version"]),
        "files": list(conflict["files"]),
        "duration_seconds": (conflict["end_us"] - conflict["start_us"]) / 1e6,
    }


def _segment_from_item(item: Mapping[str, Any]) -> VersionSegment:
    """Deserializes a segment item."""
    return VersionSegment(
        start=str(item["start"]),
        end=str(item["end"]),
        file=str(item["file"]),
        version=item["version"],
        duration_seconds=float(item["duration_seconds"]),
    )


def _read_info(ds: xr.Dataset) -> dict[str, Any]:
    """Reads the reconciliation information from a dataset's attributes."""
    raw = ds.attrs.get(INFO_ATTR)
    if raw is None:
        return {}
    try:
        info = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def _info_intervals(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Returns the valid intervals from reconciliation information."""
    return [dict(item) for item in info.get("segments", [])]


# ---------------------------------------------------------------------------
# Monthly aggregation helpers
# ---------------------------------------------------------------------------


def _resolve_data_var(ds: xr.Dataset, data_var: str | None) -> str:
    """Resolves the data variable used for monthly aggregation."""
    if data_var is not None:
        return data_var
    info = _read_info(ds)
    if "data_var" in info:
        return str(info["data_var"])

    time_name = _time_name(ds)
    bounds_keys = _bounds_keys(ds)
    candidates = [
        name
        for name in ds.data_vars
        if time_name in ds[name].dims and name not in bounds_keys
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Could not infer a single data variable. Pass `data_var=` explicitly."
        )
    return str(candidates[0])


def _time_layout(ds: xr.Dataset, data_var: str | None) -> dict[str, Any]:
    """Collects time, bounds, month and duration information for a dataset."""
    data_var = _resolve_data_var(ds, data_var)
    time_name = _time_name(ds)
    calendar = _calendar(ds[time_name].values)
    starts, ends = _interval_bounds_us(ds, time_name, calendar)

    intervals = [
        {"start_us": start, "end_us": end} for start, end in zip(starts, ends)
    ]
    layout = _layout_from_intervals(intervals, calendar)
    layout.update({"data_var": data_var, "time_name": time_name})
    return layout


def _layout_from_intervals(
    intervals: Sequence[Mapping[str, Any]], calendar: str | None
) -> dict[str, Any]:
    """Builds a per-interval layout for monthly aggregation."""
    lengths: list[float] = []
    months: dict[int, list[int]] = {}
    for idx, interval in enumerate(intervals):
        start = int(interval["start_us"])
        end = int(interval["end_us"])
        lengths.append(max(end - start, 0) / 1e6)
        months.setdefault(_month_start_us(start, calendar), []).append(idx)
    return {
        "intervals": list(intervals),
        "lengths": lengths,
        "months": months,
        "calendar": calendar,
    }


def _month_table(
    ds: xr.Dataset,
    layout: Mapping[str, Any],
    intervals: Sequence[Mapping[str, Any]] | None = None,
) -> _MonthTable:
    """Builds the set of months and their valid/conflict durations."""
    calendar = layout["calendar"]
    table = _MonthTable(calendar=calendar)

    valid_intervals = layout["intervals"] if intervals is None else intervals
    for interval in valid_intervals:
        start = int(interval["start_us"])
        end = int(interval["end_us"])
        for month, seconds in _split_by_month(start, end, calendar):
            table.valid_seconds[month] = table.valid_seconds.get(month, 0.0) + seconds

    for item in _read_info(ds).get("conflicts", []):
        start = int(item["start_us"])
        end = int(item["end_us"])
        for month, seconds in _split_by_month(start, end, calendar):
            table.conflict_seconds[month] = (
                table.conflict_seconds.get(month, 0.0) + seconds
            )
            table.valid_seconds.setdefault(month, 0.0)

    table.months = sorted(set(table.valid_seconds) | set(table.conflict_seconds))
    return table


def _split_by_month(
    start_us: int, end_us: int, calendar: str | None
) -> list[tuple[int, float]]:
    """Splits a time range into (month start, seconds) pieces."""
    pieces: list[tuple[int, float]] = []
    month = _month_start_us(start_us, calendar)
    while month < end_us:
        month_end = _next_month_us(month, calendar)
        overlap = min(end_us, month_end) - max(start_us, month)
        if overlap > 0:
            pieces.append((month, overlap / 1e6))
        month = month_end
    return pieces


def _monthly_means(
    ds: xr.Dataset, layout: Mapping[str, Any], table: _MonthTable
) -> dict[int, xr.DataArray]:
    """Computes the duration-weighted mean for each month."""
    if layout["data_var"] not in ds:
        return {}
    dv = ds[layout["data_var"]]
    time_name = layout["time_name"]
    intervals = layout["intervals"]

    numerators: dict[int, xr.DataArray] = {}
    denominators: dict[int, xr.DataArray] = {}
    for idx, interval in enumerate(intervals):
        value = dv.isel({time_name: idx}, drop=True).astype(np.float64)
        valid = value.notnull()
        for month, seconds in _split_by_month(
            int(interval["start_us"]), int(interval["end_us"]), layout["calendar"]
        ):
            if month not in numerators:
                numerators[month] = xr.zeros_like(value)
                denominators[month] = xr.zeros_like(value)
            weighted = xr.where(valid, value, 0.0) * seconds
            numerators[month] = numerators[month] + weighted
            denominators[month] = denominators[month] + xr.where(
                valid, seconds, 0.0
            )

    result: dict[int, xr.DataArray] = {}
    for month in table.months:
        if month not in numerators:
            continue
        ratio = numerators[month] / denominators[month]
        mean = ratio.where(denominators[month] > 0.0)
        mean.name = layout["data_var"]
        result[month] = mean
    return result


def _duration_array(
    layout: Mapping[str, Any], table: _MonthTable, name: str
) -> xr.DataArray:
    """Builds a per-month duration DataArray."""
    calendar = layout["calendar"]
    times = np.array(
        [_from_us(month, calendar) for month in table.months], dtype=object
    )
    values = np.array(
        [table.valid_seconds.get(month, 0.0) for month in table.months],
        dtype=np.float64,
    )
    return xr.DataArray(
        data=values,
        dims=[layout["time_name"]],
        coords={layout["time_name"]: times},
        name=name,
        attrs={"units": "s", "long_name": name.replace("_", " ")},
    )


def _assemble_monthly(
    ds: xr.Dataset,
    layout: Mapping[str, Any],
    table: _MonthTable,
    result: dict[int, xr.DataArray],
) -> xr.Dataset:
    """Assembles the output of :func:`monthly_mean`."""
    calendar = layout["calendar"]
    time_name = layout["time_name"]
    months = table.months
    times = np.array([_from_us(month, calendar) for month in months], dtype=object)

    data: dict[str, Any] = {}
    template = _monthly_template(ds, layout)
    if template is not None:
        stacked = [
            result[month] if month in result else template.copy(deep=True)
            for month in months
        ]
        if len(stacked) > 0:
            data[layout["data_var"]] = xr.concat(stacked, dim=time_name)

    data["valid_duration"] = (
        time_name,
        np.array([table.valid_seconds.get(m, 0.0) for m in months], dtype=np.float64),
    )
    data["conflict_duration"] = (
        time_name,
        np.array(
            [table.conflict_seconds.get(m, 0.0) for m in months],
            dtype=np.float64,
        ),
    )

    bounds = np.array(
        [
            [

                _from_us(m, calendar),
                _from_us(_next_month_us(m, calendar), calendar),
            ]
            for m in months
        ],
        dtype=object,
    )
    out = xr.Dataset(
        data,
        coords={
            time_name: (time_name, times),
            f"{time_name}_bnds": ((time_name, "bnds"), bounds),
        },
    )
    out[time_name].attrs.update(
        {
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": f"{time_name}_bnds",
        }
    )
    out[f"{time_name}_bnds"].attrs["xcdat_bounds"] = "True"
    out["valid_duration"].attrs["units"] = "s"
    out["conflict_duration"].attrs["units"] = "s"
    if INFO_ATTR in ds.attrs:
        out.attrs[INFO_ATTR] = ds.attrs[INFO_ATTR]
    out.attrs["operation"] = "monthly_mean"
    return out


def _monthly_template(
    ds: xr.Dataset, layout: Mapping[str, Any]
) -> xr.DataArray | None:
    """Returns a NaN-filled template for months without a mean."""
    data_var = layout["data_var"]
    if data_var not in ds:
        return None
    first = ds[data_var].isel({layout["time_name"]: 0}, drop=True)
    if first.dtype.kind not in "fiub":
        return None
    return (first.astype(np.float64) * np.nan)
