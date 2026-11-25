"""Module containing circular (directional) statistics for temporal reports.

Wind direction is an angular quantity, so it cannot be averaged like a scalar
(e.g., the arithmetic mean of 350° and 10° is 180°, which is exactly the
opposite of both observations). It also cannot be summarized with a plain
circular mean if the result must later be combined over a coarser reporting
period: a mean direction alone loses the information required to weight it
correctly against other periods.

This module computes duration-weighted circular statistics that are *additive*
and can therefore be merged exactly by duration. Each reporting period is
summarized with the three sufficient statistics below, from which the mean
direction and the concentration (resultant vector length) are derived:

- ``{var}_resultant_east``: sum of the eastward unit-vector components,
  weighted by the valid observation duration in seconds.
- ``{var}_resultant_north``: sum of the northward unit-vector components,
  weighted by the valid observation duration in seconds.
- ``{var}_weight``: total valid observation duration in seconds.

Because the sufficient statistics are sums, a monthly report can be produced by
adding the daily reports together (``combine``) and re-deriving the direction
and concentration, or directly from the raw observations (``summary``). Both
paths produce the same direction and concentration.

Observations whose time bounds span a reporting-period boundary (for example, a
six-hour observation crossing midnight) are split between the two periods in
proportion to the actual time spent in each period. Missing direction values are
excluded from both the vector sums and the total duration. A measured direction
of due north (0°) is a valid observation and is preserved as ``0``; it is not
encoded as "missing".

Two exactly opposing, equal-duration winds cancel to a zero resultant vector.
In that case the direction is undefined, so it is reported as ``NaN`` and the
concentration is reported as ``0``, even if the two winds fall in different
reporting periods. This is distinct from a genuine, well-defined due-north
direction.
"""

from typing import Literal

import cf_xarray  # noqa: F401
import cftime
import numpy as np
import xarray as xr
from xarray.coding.cftime_offsets import to_offset
from xarray.core.common import contains_cftime_datetimes

from xcdat.axis import get_dim_coords
from xcdat.bounds import _get_bounds_dim
from xcdat.dataset import _get_data_var
from xcdat.temporal import _contains_datetime_like_objects

#: Supported reporting frequencies, from finest to coarsest.
CircularFrequency = Literal["hour", "day", "month", "year"]
CIRCULAR_FREQUENCIES = ("hour", "day", "month", "year")

#: Ranking used to prevent combining a report to a finer frequency.
_FREQ_RANK = {freq: rank for rank, freq in enumerate(CIRCULAR_FREQUENCIES)}

#: Maps a reporting frequency to its calendar step and numpy unit.
_CFTIME_OFFSETS = {"hour": "h", "day": "D", "month": "MS", "year": "YS"}
_NUMPY_UNITS = {"hour": "h", "day": "D", "month": "M", "year": "Y"}

#: Reference epoch used to convert datetimes to seconds.
_EPOCH = "seconds since 1970-01-01 00:00:00"

#: Suffixes for the report variables derived from a direction variable.
_SUFFIX_DIRECTION = "direction"
_SUFFIX_CONCENTRATION = "concentration"
_SUFFIX_RESULTANT_EAST = "resultant_east"
_SUFFIX_RESULTANT_NORTH = "resultant_north"
_SUFFIX_WEIGHT = "weight"

_REPORT_ATTR = "circular_report"
_FREQ_ATTR = "circular_freq"
_SOURCE_VAR_ATTR = "circular_source_var"
_MIN_CONCENTRATION_ATTR = "circular_min_concentration"

#: Default concentration at or below which the mean direction is indeterminate.
DEFAULT_MIN_CONCENTRATION = 1e-12


@xr.register_dataset_accessor("circular")
class CircularAccessor:
    """An accessor for duration-weighted circular statistics on Datasets.

    This accessor summarizes angular (directional) data variable(s) into
    reporting periods using additive sufficient statistics, which allows daily
    reports to be merged into monthly reports without access to the original
    observations.

    The accessor operates on a data variable containing directions in degrees
    clockwise from true north (0° = north, 90° = east). The dataset's time
    coordinates must be decoded as ``np.datetime64`` or ``cftime.datetime``
    objects and must have time bounds describing the actual observation
    duration.

    Examples
    --------

    Create a daily wind direction report from hourly observations:

    >>> import xcdat  # noqa: F401
    >>>
    >>> ds_report = ds.circular.summary("wd", freq="day")

    Merge the daily report into a monthly report:

    >>> ds_monthly = ds_report.circular.combine("wd", freq="month")

    The monthly report is identical to calculating it directly from the raw
    observations:

    >>> ds_monthly_direct = ds.circular.summary("wd", freq="month")

    Parameters
    ----------
    dataset : xr.Dataset
        A Dataset object.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def summary(
        self,
        data_var: str,
        freq: CircularFrequency = "day",
        min_concentration: float = DEFAULT_MIN_CONCENTRATION,
    ) -> xr.Dataset:
        """Summarize a direction variable into duration-weighted statistics.

        Each observation is weighted by the duration (in seconds) that its time
        bounds overlap the reporting period. Observations that span a period
        boundary are split between the affected periods by their actual
        duration. Missing (``NaN``) directions contribute neither to the vector
        sums nor to the total duration.

        The returned Dataset contains the sufficient statistics required to
        merge reports by duration (``resultant_east``, ``resultant_north`` and
        ``weight``) plus the derived ``direction`` and ``concentration``.

        Parameters
        ----------
        data_var : str
            The key of the direction data variable. Values are interpreted as
            degrees clockwise from true north and are normalized modulo 360.
        freq : {"hour", "day", "month", "year"}, optional
            The reporting period frequency, by default "day".
        min_concentration : float, optional
            The concentration at or below which a reporting period's mean
            direction is considered indeterminate, by default 1e-12. This
            primarily identifies periods whose vectors cancel exactly (for
            example, equal-duration opposing winds). The direction is reported
            as ``NaN`` and the concentration as ``0`` for these periods. Must
            be in the range [0, 1].

        Returns
        -------
        xr.Dataset
            A Dataset containing the report variables for each reporting
            period, indexed by the period start time:

            - ``{data_var}_direction``: duration-weighted mean direction in
              degrees clockwise from true north, or ``NaN`` if indeterminate.
            - ``{data_var}_concentration``: resultant vector length divided by
              the total valid duration, ranging from 0 to 1.
            - ``{data_var}_resultant_east``: duration-weighted sum of eastward
              unit-vector components (seconds).
            - ``{data_var}_resultant_north``: duration-weighted sum of northward
              unit-vector components (seconds).
            - ``{data_var}_weight``: total valid observation duration (seconds).

        Raises
        ------
        KeyError
            If the data variable does not exist, or if time bounds are missing.
        TypeError
            If the data variable is not numeric or its time coordinates are not
            decoded as datetime-like objects.
        ValueError
            If ``freq`` or ``min_concentration`` is invalid, or if no
            observation has a positive duration.

        Examples
        --------
        >>> ds_daily = ds.circular.summary("wd", freq="day")
        """
        self._validate_freq(freq)
        self._validate_min_concentration(min_concentration)

        ds = self._dataset
        dv = _get_data_var(ds, data_var)
        dim = str(get_dim_coords(dv, "T").name)

        if not np.issubdtype(dv.dtype, np.number):
            raise TypeError(
                f"The data variable '{data_var}' must be numeric to calculate "
                f"circular statistics, but has dtype '{dv.dtype}'."
            )

        if not _contains_datetime_like_objects(dv[dim]):
            first = dv[dim].values[0]
            raise TypeError(
                f"The '{dim}' coordinates contain {type(first)} objects. "
                f"'{dim}' coordinates must be decoded to datetime-like objects "
                "(`np.datetime64` or `cftime.datetime`) before calculating "
                "circular statistics. Refer to `xcdat.decode_time`."
            )

        time_bounds = ds.bounds.get_bounds("T", var_key=data_var)
        bounds_dim = _get_bounds_dim(dv[dim], time_bounds)
        lower = time_bounds.isel({bounds_dim: 0})
        upper = time_bounds.isel({bounds_dim: 1})

        is_cf, calendar = _get_time_type_and_calendar(dv[dim])
        lower_s = _to_seconds(lower.values, is_cf, calendar)
        upper_s = _to_seconds(upper.values, is_cf, calendar)

        labels, edges = _period_grid(
            lower.values, upper.values, lower_s, upper_s, freq, is_cf, calendar
        )
        overlap = _overlap_matrix(lower_s, upper_s, edges)

        # Build the weighted components as xarray operations so that lazy
        # (Dask-backed) data variables remain lazy. The trigonometric terms are
        # always computed in float64: with float32 input, exactly opposing
        # directions would otherwise leave a small residual vector (~1e-8),
        # which would incorrectly appear as a well-defined direction.
        theta = dv.astype("float64") % 360.0
        valid = np.isfinite(theta)
        safe_theta = xr.where(valid, theta, 0.0)
        radians = np.deg2rad(safe_theta)

        east_component = xr.where(valid, np.sin(radians), 0.0)
        north_component = xr.where(valid, np.cos(radians), 0.0)
        valid_duration = xr.where(valid, 1.0, 0.0)

        weights = xr.DataArray(
            data=overlap,
            coords={dim: dv[dim]},
            dims=["_circular_period", dim],
            name="_circular_overlap",
        )

        resultant_east = _dot_period(east_component, weights, dim)
        resultant_north = _dot_period(north_component, weights, dim)
        weight = _dot_period(valid_duration, weights, dim)

        direction, concentration = _derive_direction_and_concentration(
            resultant_east, resultant_north, weight, min_concentration
        )

        report = xr.Dataset(
            data_vars={
                f"{data_var}_{_SUFFIX_DIRECTION}": direction,
                f"{data_var}_{_SUFFIX_CONCENTRATION}": concentration,
                f"{data_var}_{_SUFFIX_RESULTANT_EAST}": resultant_east,
                f"{data_var}_{_SUFFIX_RESULTANT_NORTH}": resultant_north,
                f"{data_var}_{_SUFFIX_WEIGHT}": weight,
            },
            attrs=dict(ds.attrs),
        )
        report = _attach_period_coordinates(
            report, dv[dim], labels, freq, is_cf, calendar
        )
        report = _set_report_attrs(
            report,
            data_var,
            freq,
            min_concentration,
            operation="circular_summary",
        )

        return report

    def combine(
        self,
        data_var: str,
        freq: CircularFrequency = "month",
    ) -> xr.Dataset:
        """Merge a report to a coarser reporting frequency by total duration.

        This method adds the sufficient statistics (``resultant_east``,
        ``resultant_north`` and ``weight``) of each source period into the
        target period and re-derives the mean direction and concentration. It
        accepts a Dataset previously produced by :meth:`summary`, so a monthly
        report can be built from daily reports alone.

        The combination is exact: for any observations summarized at a finer
        frequency, the resulting direction and concentration equal those
        computed directly at the target frequency with :meth:`summary`.

        Parameters
        ----------
        data_var : str
            The key of the original direction data variable used to create the
            report. The report variables are expected to be prefixed with this
            key (e.g., ``wd_direction`` for ``data_var="wd"``).
        freq : {"hour", "day", "month", "year"}, optional
            The target reporting frequency, by default "month". It must be the
            same as or coarser than the source report's frequency.

        Returns
        -------
        xr.Dataset
            A Dataset containing the merged report variables for each target
            period, indexed by the period start time.

        Raises
        ------
        KeyError
            If the report variables for ``data_var`` are missing.
        ValueError
            If the Dataset was not produced by :meth:`summary`, if its source
            frequency is missing or unrecognized, if the target frequency is
            finer than the source frequency, or if ``freq`` is invalid.

        Examples
        --------
        >>> ds_daily = ds.circular.summary("wd", freq="day")
        >>> ds_monthly = ds_daily.circular.combine("wd", freq="month")
        """
        self._validate_freq(freq)

        ds = self._dataset
        east_name = f"{data_var}_{_SUFFIX_RESULTANT_EAST}"
        north_name = f"{data_var}_{_SUFFIX_RESULTANT_NORTH}"
        weight_name = f"{data_var}_{_SUFFIX_WEIGHT}"

        missing = [key for key in (east_name, north_name, weight_name) if key not in ds]
        if missing:
            raise KeyError(
                "The Dataset does not contain the circular report variables "
                f"required for '{data_var}': {', '.join(missing)}. Create a "
                "report first with `Dataset.circular.summary()`."
            )

        source_freq = ds[east_name].attrs.get(_FREQ_ATTR)
        if ds[east_name].attrs.get(_REPORT_ATTR) != "True" or source_freq is None:
            raise ValueError(
                f"The variable '{east_name}' was not created by "
                "`Dataset.circular.summary()`. `combine()` requires a report "
                "containing the duration-weighted sufficient statistics."
            )
        if source_freq not in CIRCULAR_FREQUENCIES:
            raise ValueError(
                f"The report frequency attribute '{source_freq}' is not "
                f"supported. Supported frequencies include: "
                f"{list(CIRCULAR_FREQUENCIES)}."
            )
        if _FREQ_RANK[freq] < _FREQ_RANK[source_freq]:
            raise ValueError(
                f"Cannot combine a '{source_freq}' report to the finer "
                f"frequency '{freq}'. The target frequency must be the same as "
                "or coarser than the source frequency."
            )

        min_concentration = ds[east_name].attrs.get(
            _MIN_CONCENTRATION_ATTR, DEFAULT_MIN_CONCENTRATION
        )

        dim = str(get_dim_coords(ds[east_name], "T").name)
        is_cf, calendar = _get_time_type_and_calendar(ds[dim])
        labels = _truncate_times(ds[dim].values, freq, is_cf)
        labels_da = xr.DataArray(
            data=labels,
            dims=[dim],
            coords={dim: labels},
            name=dim,
            attrs=ds[dim].attrs,
        )

        resultant_east = _group_sum(ds[east_name], labels_da, dim)
        resultant_north = _group_sum(ds[north_name], labels_da, dim)
        weight = _group_sum(ds[weight_name], labels_da, dim)

        direction, concentration = _derive_direction_and_concentration(
            resultant_east, resultant_north, weight, float(min_concentration)
        )

        report = xr.Dataset(
            data_vars={
                f"{data_var}_{_SUFFIX_DIRECTION}": direction,
                f"{data_var}_{_SUFFIX_CONCENTRATION}": concentration,
                east_name: resultant_east,
                north_name: resultant_north,
                weight_name: weight,
            },
            attrs=dict(ds.attrs),
        )

        period_labels = resultant_east[dim].values
        report = _attach_period_coordinates(
            report, ds[dim], period_labels, freq, is_cf, calendar
        )
        report = _set_report_attrs(
            report,
            data_var,
            freq,
            float(min_concentration),
            operation="circular_combine",
        )

        return report

    def _validate_freq(self, freq: str):
        if freq not in CIRCULAR_FREQUENCIES:
            raise ValueError(
                f"Incorrect `freq` argument '{freq}'. Supported frequencies "
                f"include: {list(CIRCULAR_FREQUENCIES)}."
            )

    def _validate_min_concentration(self, min_concentration: float):
        if not 0.0 <= min_concentration <= 1.0:
            raise ValueError(
                "`min_concentration` must be between 0 and 1 inclusive, but "
                f"got {min_concentration}."
            )


def _get_time_type_and_calendar(time_coords: xr.DataArray) -> tuple[bool, str]:
    """Get whether the time coordinates are cftime objects and the calendar.

    Parameters
    ----------
    time_coords : xr.DataArray
        The time dimension coordinate variable.

    Returns
    -------
    tuple[bool, str]
        Whether the coordinates are ``cftime`` objects, and the calendar name.
    """
    is_cf = contains_cftime_datetimes(xr.as_variable(time_coords))

    if is_cf:
        calendar = time_coords.encoding.get("calendar")
        if calendar is None:
            calendar = getattr(time_coords.values[0], "calendar", "standard")
    else:
        calendar = time_coords.encoding.get("calendar", "standard")

    return is_cf, str(calendar)


def _to_seconds(values: np.ndarray, is_cf: bool, calendar: str) -> np.ndarray:
    """Convert datetime-like values to seconds since a fixed epoch.

    Parameters
    ----------
    values : np.ndarray
        An array of ``np.datetime64`` or ``cftime.datetime`` values.
    is_cf : bool
        Whether the values are ``cftime`` objects.
    calendar : str
        The CF calendar name.

    Returns
    -------
    np.ndarray
        A float64 array of seconds since 1970-01-01 00:00:00. NaT/NaN entries
        are returned as ``NaN``.
    """
    if is_cf:
        values = np.asarray(values, dtype=object)
        missing = np.array([value is None for value in values.ravel()]).reshape(
            values.shape
        )
        seconds = np.full(values.shape, np.nan, dtype=np.float64)
        if not missing.all():
            seconds[~missing] = np.asarray(
                cftime.date2num(values[~missing], _EPOCH, calendar=calendar),
                dtype=np.float64,
            )
        return seconds

    datetimes = np.asarray(values).astype("datetime64[us]")
    missing = np.isnat(datetimes)
    seconds = datetimes.astype(np.int64).astype(np.float64) / 1e6
    seconds[missing] = np.nan
    return seconds


def _truncate_scalar(value, freq: str, is_cf: bool):
    """Truncate a single datetime-like value to the start of its period."""
    if is_cf:
        replacements = {"minute": 0, "second": 0, "microsecond": 0}
        if freq in ("day", "month", "year"):
            replacements["hour"] = 0
        if freq in ("month", "year"):
            replacements["day"] = 1
        if freq == "year":
            replacements["month"] = 1
        return value.replace(**replacements)

    unit = _NUMPY_UNITS[freq]
    return np.asarray([value]).astype(f"datetime64[{unit}]")[0]


def _truncate_times(values: np.ndarray, freq: str, is_cf: bool) -> np.ndarray:
    """Truncate an array of datetime-like values to the start of each period."""
    truncated = [_truncate_scalar(value, freq, is_cf) for value in np.asarray(values)]

    if is_cf:
        return np.array(truncated, dtype=object)

    # Preserve the input precision (e.g., datetime64[ns]).
    dtype = np.asarray(values).dtype
    return np.array(truncated).astype(dtype)


def _add_period(value, freq: str, is_cf: bool):
    """Return the start of the period immediately following ``value``."""
    if is_cf:
        return value + to_offset(_CFTIME_OFFSETS[freq])

    # Cast to the period's unit before adding, because NumPy cannot add a
    # month/year timedelta to a datetime64 in a finer unit (e.g., [ns]).
    unit = _NUMPY_UNITS[freq]
    truncated = np.asarray(value).astype(f"datetime64[{unit}]")

    return (truncated + np.timedelta64(1, unit))[()]


def _period_grid(
    lower_values: np.ndarray,
    upper_values: np.ndarray,
    lower_seconds: np.ndarray,
    upper_seconds: np.ndarray,
    freq: str,
    is_cf: bool,
    calendar: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the reporting-period start labels and their boundary seconds.

    The grid starts at the period containing the earliest observation and ends
    at the period containing the latest observation.

    Parameters
    ----------
    lower_values : np.ndarray
        The lower time bounds.
    upper_values : np.ndarray
        The upper time bounds.
    lower_seconds : np.ndarray
        The lower time bounds in seconds.
    upper_seconds : np.ndarray
        The upper time bounds in seconds.
    freq : str
        The reporting frequency.
    is_cf : bool
        Whether the time values are ``cftime`` objects.
    calendar : str
        The CF calendar name.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The period start labels and the period boundary seconds (one more than
        the number of periods).

    Raises
    ------
    ValueError
        If there are no observations with a finite, positive duration.
    """
    valid = np.isfinite(lower_seconds) & np.isfinite(upper_seconds)
    valid &= upper_seconds > lower_seconds

    if not valid.any():
        raise ValueError(
            "No observations have a finite, positive duration based on their "
            "time bounds, so a circular report cannot be created."
        )

    valid_indices = np.nonzero(valid)[0]
    first_idx = valid_indices[np.argmin(lower_seconds[valid_indices])]
    last_idx = valid_indices[np.argmax(upper_seconds[valid_indices])]

    start = _truncate_scalar(lower_values[first_idx], freq, is_cf)
    labels = _period_labels(start, upper_values[last_idx], freq, is_cf)

    if labels.size == 0:
        raise ValueError(
            "No reporting periods were found for the time bounds, so a "
            "circular report cannot be created."
        )

    next_label = _add_period(labels[-1], freq, is_cf)
    edges = _to_seconds(
        np.array([*labels, next_label], dtype=object if is_cf else None),
        is_cf,
        calendar,
    )

    return labels, np.asarray(edges, dtype=np.float64)


def _period_labels(
    start,
    end_value,
    freq: str,
    is_cf: bool,
) -> np.ndarray:
    """List the start of every reporting period from ``start`` up to ``end``.

    The period containing ``end_value`` is included, while a period starting
    exactly at ``end_value`` is excluded (the range is half-open).

    Parameters
    ----------
    start :
        The first period start, already truncated to the period.
    end_value :
        The latest upper time bound.
    freq : str
        The reporting frequency.
    is_cf : bool
        Whether the values are ``cftime`` objects.

    Returns
    -------
    np.ndarray
        The period start labels.
    """
    if is_cf:
        # Iterate with cftime offsets rather than ``xr.date_range`` so this
        # works with the minimum supported Xarray version (2024.03.0).
        offset = to_offset(_CFTIME_OFFSETS[freq])
        labels = []
        current = start
        while current < end_value:
            labels.append(current)
            current = current + offset
        return np.array(labels, dtype=object)

    unit = _NUMPY_UNITS[freq]
    unit_dtype = np.dtype(f"datetime64[{unit}]")
    start_count = np.asarray(start).astype(unit_dtype).astype(np.int64)
    truncated_end = np.asarray(end_value).astype(unit_dtype)
    end_count = truncated_end.astype(np.int64)

    # Include the period containing ``end_value``, unless ``end_value`` falls
    # exactly on a period boundary (the range is half-open on the end).
    if np.asarray(end_value) > truncated_end:
        end_count += 1

    counts = np.arange(start_count, end_count)

    return counts.astype(unit_dtype)


def _overlap_matrix(
    lower_seconds: np.ndarray, upper_seconds: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Calculate the duration each observation spends in each period.

    Parameters
    ----------
    lower_seconds : np.ndarray
        The lower time bounds in seconds.
    upper_seconds : np.ndarray
        The upper time bounds in seconds.
    edges : np.ndarray
        The period boundary seconds, with one more entry than the periods.

    Returns
    -------
    np.ndarray
        A ``(n_periods, n_observations)`` array of overlap durations in
        seconds. Invalid observations have zero overlap in every period.
    """
    n_periods = len(edges) - 1
    n_obs = lower_seconds.size
    overlap = np.zeros((n_periods, n_obs), dtype=np.float64)

    valid = np.isfinite(lower_seconds) & np.isfinite(upper_seconds)
    valid &= upper_seconds > lower_seconds
    indices = np.nonzero(valid)[0]

    if indices.size == 0:
        return overlap

    lower = lower_seconds[indices]
    upper = upper_seconds[indices]

    start_period = np.searchsorted(edges, lower, side="right") - 1
    end_period = np.searchsorted(edges, upper, side="left") - 1
    start_period = np.clip(start_period, 0, n_periods - 1)
    end_period = np.clip(end_period, 0, n_periods - 1)

    # Observations contained within a single period.
    same_period = start_period == end_period
    same_idx = indices[same_period]
    overlap[start_period[same_period], same_idx] = upper[same_period] - lower[
        same_period
    ]

    # Observations that cross one or more period boundaries.
    cross_idx = np.nonzero(~same_period)[0]
    for position in cross_idx:
        obs_idx = indices[position]
        for period in range(start_period[position], end_period[position] + 1):
            overlap_seconds = min(upper[position], edges[period + 1]) - max(
                lower[position], edges[period]
            )
            if overlap_seconds > 0.0:
                overlap[period, obs_idx] += overlap_seconds

    return overlap


def _dot_period(
    data_var: xr.DataArray, weights: xr.DataArray, dim: str
) -> xr.DataArray:
    """Sum a data variable over time weighted by duration, per period."""
    result = xr.dot(data_var, weights, dim=dim).transpose("_circular_period", ...)
    result = result.rename({"_circular_period": dim})

    return result


def _group_sum(
    data_var: xr.DataArray, labels: xr.DataArray, dim: str
) -> xr.DataArray:
    """Sum a data variable into groups labeled by the target period starts."""
    attrs = data_var.attrs
    labeled = data_var.assign_coords({dim: labels})
    grouped = labeled.groupby(dim).sum()
    grouped.attrs = attrs

    return grouped


def _derive_direction_and_concentration(
    resultant_east: xr.DataArray,
    resultant_north: xr.DataArray,
    weight: xr.DataArray,
    min_concentration: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Derive the mean direction and concentration from resultant vectors.

    The direction is ``NaN`` both where there are no valid observations and
    where the vectors cancel (indeterminate). The concentration distinguishes
    the two: it is ``NaN`` where there are no valid observations and ``0`` where
    the direction is indeterminate.

    Parameters
    ----------
    resultant_east : xr.DataArray
        Duration-weighted sum of eastward unit-vector components.
    resultant_north : xr.DataArray
        Duration-weighted sum of northward unit-vector components.
    weight : xr.DataArray
        Total valid observation duration.
    min_concentration : float
        Concentration at or below which the direction is indeterminate.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        The mean direction in degrees clockwise from true north (``NaN`` where
        indeterminate or where no valid observations exist) and the
        concentration (0 where the vectors cancel, ``NaN`` where no valid
        observations exist).
    """
    defined = weight > 0.0
    magnitude = np.hypot(resultant_east, resultant_north)

    safe_weight = xr.where(defined, weight, 1.0)
    concentration = xr.where(defined, magnitude / safe_weight, np.nan)
    indeterminate = defined & (concentration <= min_concentration)
    concentration = xr.where(indeterminate, 0.0, concentration)

    radians = np.arctan2(resultant_east, resultant_north)
    direction = np.degrees(radians)
    direction = xr.where(direction < 0.0, direction + 360.0, direction)
    # Normalize the tiny positive residue of a negative zero angle (which would
    # otherwise read as 359.99999999999994) back to true north.
    direction = xr.where(
        np.isclose(direction, 360.0, rtol=0.0, atol=1e-9), 0.0, direction
    )
    direction = xr.where(defined & ~indeterminate, direction, np.nan)

    # ``xr.where`` drops coordinate attributes when the condition combines
    # boolean DataArrays (e.g., ``defined & ~indeterminate``). Restore the
    # coordinate metadata so that non-time coordinates such as latitude and
    # longitude keep their attributes in the report.
    other_coords = {
        name: coord for name, coord in resultant_east.coords.items() if name != "time"
    }
    direction = direction.assign_coords(other_coords)
    concentration = concentration.assign_coords(other_coords)

    return direction, concentration


def _attach_period_coordinates(
    report: xr.Dataset,
    source_time_coords: xr.DataArray,
    labels: np.ndarray,
    freq: str,
    is_cf: bool,
    calendar: str,
) -> xr.Dataset:
    """Attach period start coordinates and period bounds to a report."""
    dim = str(source_time_coords.name)
    bounds_name = f"{dim}_bnds"

    if is_cf:
        upper_labels = np.array(
            [_add_period(label, freq, is_cf) for label in labels], dtype=object
        )
        labels = np.array(labels, dtype=object)
    else:
        dtype = np.asarray(source_time_coords.values).dtype
        labels = np.asarray(labels).astype(dtype)
        upper_labels = np.array(
            [_add_period(label, freq, is_cf) for label in labels]
        ).astype(dtype)

    time_attrs = dict(source_time_coords.attrs)
    time_attrs["bounds"] = bounds_name
    time_attrs.setdefault("axis", "T")

    time_coords = xr.DataArray(
        data=labels,
        dims=[dim],
        attrs=time_attrs,
        name=dim,
    )
    if is_cf:
        time_coords.encoding["calendar"] = calendar
    else:
        time_coords.encoding = dict(source_time_coords.encoding)

    bounds = xr.DataArray(
        name=bounds_name,
        data=np.stack([labels, upper_labels], axis=1),
        dims=[dim, "bnds"],
        attrs={"xcdat_bounds": "True"},
    )

    # Assign the bounds first, then the period coordinates, so that the
    # bounds' unlabeled time coordinate cannot overwrite the coordinate
    # attributes.
    report[bounds_name] = bounds
    report = report.assign_coords({dim: time_coords})

    return report


def _set_report_attrs(
    report: xr.Dataset,
    data_var: str,
    freq: str,
    min_concentration: float,
    operation: str,
) -> xr.Dataset:
    """Set attributes marking the Dataset as a circular report."""
    report.attrs[_REPORT_ATTR] = "True"
    report.attrs[_FREQ_ATTR] = freq
    report.attrs[_SOURCE_VAR_ATTR] = data_var
    report.attrs[_MIN_CONCENTRATION_ATTR] = float(min_concentration)

    common_attrs = {
        _REPORT_ATTR: "True",
        _FREQ_ATTR: freq,
        _SOURCE_VAR_ATTR: data_var,
        _MIN_CONCENTRATION_ATTR: float(min_concentration),
        "operation": operation,
    }

    long_names = {
        _SUFFIX_DIRECTION: (
            "duration-weighted mean direction in degrees clockwise from true "
            "north; NaN where indeterminate"
        ),
        _SUFFIX_CONCENTRATION: (
            "resultant vector length divided by total valid duration; 0 where "
            "the direction is indeterminate"
        ),
        _SUFFIX_RESULTANT_EAST: (
            "duration-weighted sum of eastward unit-vector components"
        ),
        _SUFFIX_RESULTANT_NORTH: (
            "duration-weighted sum of northward unit-vector components"
        ),
        _SUFFIX_WEIGHT: "total valid observation duration",
    }
    units = {
        _SUFFIX_DIRECTION: "degrees",
        _SUFFIX_CONCENTRATION: "1",
        _SUFFIX_RESULTANT_EAST: "s",
        _SUFFIX_RESULTANT_NORTH: "s",
        _SUFFIX_WEIGHT: "s",
    }

    for suffix, long_name in long_names.items():
        key = f"{data_var}_{suffix}"
        if key not in report:
            continue
        report[key].attrs.update(common_attrs)
        report[key].attrs["long_name"] = long_name
        report[key].attrs["units"] = units[suffix]

    return report
