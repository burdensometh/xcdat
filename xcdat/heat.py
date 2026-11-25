"""Module containing heat exposure statistics.

Heat exposure measures the fraction of weighted space-time volume in which a
temperature variable exceeds a threshold. The statistic is computed from two
additive sufficient statistics:

- ``exposed``: the weight (temporal duration multiplied by grid-cell area)
  where the variable is at or above the threshold; and
- ``coverage``: the total weight over valid (non-missing) observations.

The exposure fraction is ``exposed / coverage``. Because both components are
simple weighted sums, statistics computed over different regions or adjacent
time periods can be merged and produce exactly the same fraction as a single
calculation over the union of the observations. Grid cells that are only
partially covered by a region contribute proportionally to their overlapping
area, time steps that cross a grouping boundary contribute proportionally to
their overlapping duration, and missing observations contribute to neither
the numerator nor the denominator.
"""

from dataclasses import dataclass
from typing import Literal, get_args

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from xarray.core.common import is_np_datetime_like

from xcdat.axis import get_dim_keys
from xcdat.dataset import _get_data_var

#: Type alias for supported temporal grouping frequencies.
HeatFrequency = Literal["year", "month"]
#: Tuple of supported temporal grouping frequencies.
HEAT_FREQUENCIES = get_args(HeatFrequency)

#: Type alias for supported temperature units.
TemperatureUnit = Literal["degC", "K"]

#: Offset in Kelvin between the Celsius and Kelvin temperature scales.
KELVIN_OFFSET = 273.15

#: CF-compliant unit strings recognized as Celsius.
_CELSIUS_UNITS = frozenset(
    {
        "degree_celsius",
        "degrees_celsius",
        "degrees_c",
        "degree_c",
        "degc",
        "celsius",
        "c",
    }
)
#: CF-compliant unit strings recognized as Kelvin.
_KELVIN_UNITS = frozenset(
    {
        "kelvin",
        "degrees_kelvin",
        "degree_kelvin",
        "degrees_k",
        "degk",
        "k",
    }
)


@dataclass
class HeatExposureStats:
    """Sufficient statistics for a heat exposure calculation.

    ``exposed`` and ``coverage`` share the same dimensions. Every value of
    ``exposed`` is bounded by the corresponding value of ``coverage``. Both
    arrays are additive: merging two statistics amounts to summing their
    components (after alignment), which makes partial results over different
    regions or adjacent time periods combinable under a single definition.

    Parameters
    ----------
    exposed : xr.DataArray
        Weighted space-time volume in which the temperature variable is at or
        above the threshold.
    coverage : xr.DataArray
        Total weighted space-time volume over valid observations.
    threshold : float
        Threshold value used to define heat exposure, in units of
        ``threshold_units``.
    threshold_units : str
        Units of ``threshold`` (``"degC"`` or ``"K"``).
    data_var : str
        Name of the data variable the statistics were computed from.
    freq : str | None
        Temporal grouping frequency (``"year"``, ``"month"``) or ``None`` when
        no temporal grouping was applied.
    """

    exposed: xr.DataArray
    coverage: xr.DataArray
    threshold: float
    threshold_units: TemperatureUnit
    data_var: str
    freq: HeatFrequency | None

    @property
    def fraction(self) -> xr.DataArray:
        """Fraction of the valid weighted volume above the threshold.

        Returns
        -------
        xr.DataArray
            ``exposed / coverage``. Values are NaN where ``coverage`` is zero
            (e.g. an entirely missing region), preserving them as unknown
            rather than reporting zero exposure.
        """
        # Replace zero coverage with one before dividing so the operation
        # never triggers a divide-by-zero (including on lazily-evaluated Dask
        # arrays, which do not inherit np.errstate); those cells are then
        # overwritten with NaN.
        safe_coverage = xr.where(
            self.coverage > 0.0, self.coverage, 1.0, keep_attrs=False
        )
        fraction = xr.where(
            self.coverage > 0.0,
            self.exposed / safe_coverage,
            np.nan,
            keep_attrs=False,
        )
        fraction.name = f"{self.data_var}_heat_exposure_fraction"
        fraction.attrs = {
            "long_name": (
                f"Fraction of weighted volume with {self.data_var} >= "
                f"{self.threshold} {self.threshold_units}"
            ),
            "units": "1",
            "threshold": self.threshold,
            "threshold_units": self.threshold_units,
        }
        return fraction

    def to_dataset(self) -> xr.Dataset:
        """Package the statistics into a Dataset.

        Returns
        -------
        xr.Dataset
            Dataset containing the ``exposed``, ``coverage`` and ``fraction``
            arrays along with descriptive global attributes.
        """
        exposed = self.exposed.copy()
        coverage = self.coverage.copy()
        exposed.name = f"{self.data_var}_heat_exposed"
        coverage.name = f"{self.data_var}_heat_coverage"
        exposed.attrs = {
            "long_name": (
                f"Weighted volume with {self.data_var} >= "
                f"{self.threshold} {self.threshold_units}"
            ),
            "threshold": self.threshold,
            "threshold_units": self.threshold_units,
        }
        coverage.attrs = {
            "long_name": f"Total valid weighted volume for {self.data_var}",
            "threshold": self.threshold,
            "threshold_units": self.threshold_units,
        }

        ds = xr.Dataset(
            {
                exposed.name: exposed,
                coverage.name: coverage,
                self.fraction.name: self.fraction,
            }
        )
        ds.attrs = {
            "heat_exposure_data_var": self.data_var,
            "heat_exposure_threshold": self.threshold,
            "heat_exposure_threshold_units": self.threshold_units,
            "heat_exposure_freq": "" if self.freq is None else self.freq,
        }
        return ds

    def merge(self, other: "HeatExposureStats") -> "HeatExposureStats":
        """Merge with statistics over another region or time period.

        The statistics must share the same threshold, threshold units, source
        variable name and grouping frequency. Missing components (locations or
        time groups absent from one operand) are treated as zero volume, so
        disjoint regions and adjacent time periods combine additively.

        Parameters
        ----------
        other : HeatExposureStats
            Statistics to merge with.

        Returns
        -------
        HeatExposureStats
            Combined statistics whose fraction equals that of a direct
            calculation over the union of the underlying observations.

        Raises
        ------
        ValueError
            If the statistics were computed under different thresholds,
            threshold units, source variables or grouping frequencies.
        """
        if self.threshold_units != other.threshold_units:
            raise ValueError(
                "Cannot merge heat exposure statistics with different "
                f"threshold units: {self.threshold_units!r} vs "
                f"{other.threshold_units!r}."
            )
        if self.threshold != other.threshold:
            raise ValueError(
                "Cannot merge heat exposure statistics with different "
                f"thresholds: {self.threshold} vs {other.threshold}."
            )
        if self.data_var != other.data_var:
            raise ValueError(
                "Cannot merge heat exposure statistics computed from different "
                f"variables: {self.data_var!r} vs {other.data_var!r}."
            )
        if self.freq != other.freq:
            raise ValueError(
                "Cannot merge heat exposure statistics with different grouping "
                f"frequencies: {self.freq!r} vs {other.freq!r}."
            )

        # Align on the union of coordinates; absent components carry no volume.
        exposed_a, exposed_b = xr.align(
            self.exposed, other.exposed, join="outer", fill_value=0.0
        )
        coverage_a, coverage_b = xr.align(
            self.coverage, other.coverage, join="outer", fill_value=0.0
        )

        return HeatExposureStats(
            exposed=exposed_a + exposed_b,
            coverage=coverage_a + coverage_b,
            threshold=self.threshold,
            threshold_units=self.threshold_units,
            data_var=self.data_var,
            freq=self.freq,
        )


@xr.register_dataset_accessor("heat")
class HeatAccessor:
    """An accessor class that provides heat exposure statistics on Datasets.

    Examples
    --------
    Compute the fraction of area-time above 30 degC, split by month:

    >>> stats = ds.heat.exposure("tas", threshold=30.0, freq="month")
    >>> stats.fraction

    Merge partial statistics computed over two adjacent months:

    >>> jan = ds.sel(time=slice("2000-01", "2000-01")).heat.exposure(
    ...     "tas", threshold=30.0, freq="month"
    ... )
    >>> feb = ds.sel(time=slice("2000-02", "2000-02")).heat.exposure(
    ...     "tas", threshold=30.0, freq="month"
    ... )
    >>> combined = jan.merge(feb)
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    def exposure(
        self,
        data_var: str,
        threshold: float,
        threshold_units: TemperatureUnit = "degC",
        freq: HeatFrequency | None = None,
        lat_bounds: tuple[float, float] | None = None,
        lon_bounds: tuple[float, float] | None = None,
        weights: xr.DataArray | None = None,
    ) -> HeatExposureStats:
        """Compute heat exposure sufficient statistics.

        For every time step and grid cell, the cell is flagged when
        ``data_var >= threshold`` (the threshold is converted to the
        variable's units when needed). The flag is then integrated using
        weights that combine:

        - temporal duration from the time bounds, split proportionally when a
          time step crosses a grouping boundary (e.g. a month boundary); and
        - grid-cell area (cosine of latitude times longitude span), scaled
          proportionally when a cell is only partially inside the selected
          region.

        Missing observations (NaN) contribute to neither the exposed
        numerator nor the coverage denominator. Locations or time groups with
        no valid observations have zero coverage and report a NaN fraction.

        Parameters
        ----------
        data_var : str
            Name of the temperature data variable.
        threshold : float
            Heat exposure threshold in units of ``threshold_units``.
        threshold_units : {"degC", "K"}, optional
            Units of ``threshold``, by default ``"degC"``. The threshold is
            converted to the data variable's units before comparison so that
            Celsius and Kelvin inputs yield identical results after an
            equivalent threshold conversion.
        freq : {"year", "month"} | None, optional
            Temporal grouping frequency, by default None. When ``"month"`` or
            ``"year"``, statistics are reported per calendar month or year and
            time steps crossing a boundary are split by overlapping duration.
            When None, a single statistic over the full time dimension is
            returned.
        lat_bounds : tuple[float, float] | None, optional
            Regional latitude bounds ``(lower, upper)``. Cells partially
            inside the region receive partial area weight, by default None.
        lon_bounds : tuple[float, float] | None, optional
            Regional longitude bounds ``(lower, upper)``. The lower bound may
            exceed the upper bound to select a region crossing the prime
            meridian, by default None.
        weights : xr.DataArray | None, optional
            Custom spatial area weights with the same spatial dimensions as
            ``data_var``. When None, weights are generated from the latitude
            and longitude bounds, by default None.

        Returns
        -------
        HeatExposureStats
            Additive exposed/coverage statistics, including the
            :attr:`~HeatExposureStats.fraction` property.

        Raises
        ------
        KeyError
            If ``data_var`` does not exist in the Dataset.
        ValueError
            If ``freq`` is unsupported, ``threshold_units`` is unsupported, or
            the data variable lacks a recognizable temperature units
            attribute.
        """
        if freq is not None and freq not in HEAT_FREQUENCIES:
            raise ValueError(
                f"Unsupported freq {freq!r}. Expected one of "
                f"{HEAT_FREQUENCIES} or None."
            )

        ds = self._dataset
        dv = _get_data_var(ds, data_var)
        time_dim = str(get_dim_keys(dv, "T"))

        effective_threshold = _convert_threshold(
            threshold, threshold_units, dv.attrs.get("units")
        )

        spatial_weights = _get_spatial_weights(
            ds, dv, lat_bounds, lon_bounds, weights
        )

        # Build time fragments. Each fragment carries its source time-step
        # index and group label on the shared fragment dimension. A step
        # crossing a group boundary produces two fragments with duration
        # weights proportional to the overlap on each side.
        fragment_dim = f"{time_dim}_fragment"
        time_weights, group_labels, source_index = _build_time_fragments(
            ds, dv, freq, fragment_dim
        )

        # Expand the data onto the fragment dimension via explicit source
        # indices, so data, weights and labels align without inference.
        # Indexing with a DataArray renames the dimension to the fragment
        # dimension; the original time coordinate is dropped because fragment
        # rows are not indexed by it, and grouped output receives a fresh
        # group-label coordinate.
        dv_fragments = dv.isel({time_dim: source_index})
        dv_fragments = dv_fragments.drop_vars(time_dim, errors="ignore")

        # Combined weight of every fragment: duration x area.
        combined_weights = time_weights * spatial_weights

        valid = dv_fragments.notnull()
        exposed_mask = valid & (dv_fragments >= effective_threshold)

        coverage_weights = xr.where(valid, combined_weights, 0.0)
        exposed_weights = xr.where(exposed_mask, combined_weights, 0.0)

        if freq is None:
            exposed = exposed_weights.sum(dim=fragment_dim, skipna=False)
            coverage = coverage_weights.sum(dim=fragment_dim, skipna=False)
        else:
            exposed = _group_fragments(
                exposed_weights, group_labels, time_dim
            )
            coverage = _group_fragments(
                coverage_weights, group_labels, time_dim
            )

        exposed.name = f"{data_var}_heat_exposed"
        coverage.name = f"{data_var}_heat_coverage"

        return HeatExposureStats(
            exposed=exposed,
            coverage=coverage,
            threshold=threshold,
            threshold_units=threshold_units,
            data_var=data_var,
            freq=freq,
        )


def _group_fragments(
    weighted: xr.DataArray, group_labels: xr.DataArray, time_dim: str
) -> xr.DataArray:
    """Aggregate weighted fragments by group label.

    Grouping is performed manually rather than through ``DataArray.groupby``
    so it works uniformly for cftime object calendars and numpy datetime
    calendars.

    Parameters
    ----------
    weighted : xr.DataArray
        Weighted data on a fragment dimension and any spatial dimensions.
    group_labels : xr.DataArray
        Group-start datetime for every fragment, on the same fragment
        dimension.
    time_dim : str
        Name of the time dimension to restore on the grouped output.

    Returns
    -------
    xr.DataArray
        Weighted sum per group with a CF-annotated datetime time coordinate.
    """
    fragment_dim = group_labels.dims[0]
    labels = np.asarray(group_labels.values)
    unique_labels, inverse = np.unique(labels, return_inverse=True)

    groups = []
    for code in range(len(unique_labels)):
        member_index = np.where(inverse == code)[0]
        groups.append(
            weighted.isel({fragment_dim: member_index}).sum(
                dim=fragment_dim, skipna=False
            )
        )

    grouped = xr.concat(groups, dim=time_dim)
    grouped = grouped.assign_coords(
        {
            time_dim: xr.DataArray(
                unique_labels,
                dims=[time_dim],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                },
            )
        }
    )
    return grouped


def _build_time_fragments(
    ds: xr.Dataset,
    dv: xr.DataArray,
    freq: HeatFrequency | None,
    fragment_dim: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build duration weights, group labels and source indices per fragment.

    Each time step contributes one fragment, or two fragments when its bounds
    cross a grouping boundary. The two fragments of a split step carry weights
    proportional to the duration overlapping each side of the boundary, and
    both reference the same source time-step index.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing time bounds.
    dv : xr.DataArray
        Data variable defining the time dimension.
    freq : {"year", "month"} | None
        Grouping frequency, or None for a single group.
    fragment_dim : str
        Name for the output fragment dimension.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        ``(weights, labels, source_index)``, all on ``fragment_dim``. Weights
        are absolute durations in seconds (not normalized by total
        duration), matching the unnormalized absolute area weights so partial
        statistics remain additive under merging; labels hold each fragment's
        group-start datetime; source_index gives the originating time-step
        index.
    """
    time_dim = str(get_dim_keys(dv, "T"))
    time_bounds = ds.bounds.get_bounds("T", var_key=str(dv.name))
    bounds_dim = _get_bounds_dim(dv[time_dim], time_bounds)

    lowers = time_bounds.isel({bounds_dim: 0})
    uppers = time_bounds.isel({bounds_dim: 1})
    durations = _timedelta_to_float(uppers - lowers)

    weight_values: list[float] = []
    label_values: list = []
    source_values: list[int] = []

    if freq is None:
        for index in range(durations.sizes[time_dim]):
            weight_values.append(
                float(durations.isel({time_dim: index}, drop=True).values)
            )
            label_values.append(
                _scalar(lowers.isel({time_dim: index}, drop=True))
            )
            source_values.append(index)
    else:
        starts = _group_starts(lowers, freq)
        next_starts = _group_starts(uppers, freq)
        same_group = starts == next_starts
        boundary = next_starts
        split_first_duration = _timedelta_to_float(boundary - lowers)
        split_second_duration = _timedelta_to_float(uppers - boundary)

        for index in range(durations.sizes[time_dim]):
            if bool(same_group.isel({time_dim: index})):
                # The step stays inside one group: its full duration belongs
                # to that group.
                source_values.append(index)
                weight_values.append(
                    float(durations.isel({time_dim: index}, drop=True).values)
                )
                label_values.append(
                    _scalar(starts.isel({time_dim: index}, drop=True))
                )
            else:
                # The step reaches into the next group: split at the boundary
                # into a first fragment (start group) and a second fragment
                # (next group). A zero-duration tail produced when the upper
                # bound lands exactly on a boundary is omitted because it
                # contributes no volume.
                source_values.append(index)
                weight_values.append(
                    float(
                        split_first_duration.isel(
                            {time_dim: index}, drop=True
                        ).values
                    )
                )
                label_values.append(
                    _scalar(starts.isel({time_dim: index}, drop=True))
                )

                second = float(
                    split_second_duration.isel(
                        {time_dim: index}, drop=True
                    ).values
                )
                if second > 0.0:
                    source_values.append(index)
                    weight_values.append(second)
                    label_values.append(
                        _scalar(next_starts.isel({time_dim: index}, drop=True))
                    )

    weights = xr.DataArray(
        np.array(weight_values, dtype=np.float64),
        dims=[fragment_dim],
    )
    labels = xr.DataArray(
        np.array(label_values, dtype=object), dims=[fragment_dim]
    )
    source_index = xr.DataArray(
        np.array(source_values, dtype=np.int64), dims=[fragment_dim]
    )
    return weights, labels, source_index


def _get_bounds_dim(time_coords: xr.DataArray, time_bounds: xr.DataArray) -> str:
    """Get the bounds (second) dimension name of a bounds variable.

    Parameters
    ----------
    time_coords : xr.DataArray
        Time coordinate variable.
    time_bounds : xr.DataArray
        Time bounds variable with two dimensions.

    Returns
    -------
    str
        Name of the bounds dimension (the dimension other than the time
        dimension).
    """
    time_dim = str(time_coords.dims[0])
    bounds_dims = [dim for dim in time_bounds.dims if dim != time_dim]
    return str(bounds_dims[0])


def _timedelta_to_float(delta: xr.DataArray) -> xr.DataArray:
    """Convert a timedelta-like DataArray to elapsed seconds as float64.

    Works for both ``np.timedelta64`` (numpy datetime calendars) and object
    arrays of ``datetime.timedelta`` (cftime calendars). Seconds are used
    rather than nanoseconds to keep accumulated area-time magnitudes moderate.

    Parameters
    ----------
    delta : xr.DataArray
        DataArray of timedeltas.

    Returns
    -------
    xr.DataArray
        Float64 DataArray of elapsed seconds.
    """
    if np.issubdtype(delta.dtype, np.timedelta64):
        return delta.astype("timedelta64[ns]").astype(np.float64) / 1.0e9

    seconds = np.vectorize(lambda td: td.total_seconds())(
        np.asarray(delta.values, dtype=object)
    )
    return xr.DataArray(
        seconds, dims=delta.dims, coords=delta.coords
    )


def _group_starts(time_values: xr.DataArray, freq: HeatFrequency) -> xr.DataArray:
    """Return the first instant of each value's group.

    Parameters
    ----------
    time_values : xr.DataArray
        Datetime-like coordinate values.
    freq : {"year", "month"}
        Grouping frequency.

    Returns
    -------
    xr.DataArray
        Group-start datetimes with the same shape. For numpy datetime calendars
        the result is ``datetime64``; for cftime calendars the result is cftime
        datetimes at the first of the month/year.
    """
    if is_np_datetime_like(xr.as_variable(time_values).dtype):
        resolution = "datetime64[M]" if freq == "month" else "datetime64[Y]"
        return xr.DataArray(
            time_values.values.astype(resolution),
            dims=time_values.dims,
            coords=time_values.coords,
        )

    if freq == "month":
        replaced = np.vectorize(lambda dt: dt.replace(day=1))(
            np.asarray(time_values.values, dtype=object)
        )
    else:
        replaced = np.vectorize(lambda dt: dt.replace(month=1, day=1))(
            np.asarray(time_values.values, dtype=object)
        )
    return xr.DataArray(
        replaced, dims=time_values.dims, coords=time_values.coords
    )


def _convert_threshold(
    threshold: float,
    threshold_units: TemperatureUnit,
    data_units: str | None,
) -> float:
    """Convert a threshold to the units of the data variable.

    Parameters
    ----------
    threshold : float
        Threshold value in ``threshold_units``.
    threshold_units : {"degC", "K"}
        Units of ``threshold``.
    data_units : str | None
        Units attribute of the data variable.

    Returns
    -------
    float
        Threshold value expressed in ``data_units``.

    Raises
    ------
    ValueError
        If ``threshold_units`` is unsupported or ``data_units`` cannot be
        interpreted as a temperature scale.
    """
    if threshold_units not in get_args(TemperatureUnit):
        raise ValueError(
            f"Unsupported threshold_units {threshold_units!r}. Expected "
            "'degC' or 'K'."
        )
    if data_units is None:
        raise ValueError(
            "The data variable does not have a 'units' attribute, so the "
            "threshold cannot be converted. Set a CF-compliant temperature "
            "units attribute ('degC' or 'K') on the variable."
        )

    normalized = data_units.strip().lower().replace(" ", "_")
    if normalized in _CELSIUS_UNITS:
        data_scale: TemperatureUnit = "degC"
    elif normalized in _KELVIN_UNITS:
        data_scale = "K"
    else:
        raise ValueError(
            f"Unrecognized temperature units {data_units!r}. Expected a "
            "CF-compliant Celsius (e.g. 'degrees_C') or Kelvin (e.g. 'K') "
            "units string."
        )

    if threshold_units == data_scale:
        return float(threshold)
    if threshold_units == "K" and data_scale == "degC":
        return float(threshold) - KELVIN_OFFSET
    # threshold in degC, data in K
    return float(threshold) + KELVIN_OFFSET


def _get_spatial_weights(
    ds: xr.Dataset,
    dv: xr.DataArray,
    lat_bounds: tuple[float, float] | None,
    lon_bounds: tuple[float, float] | None,
    custom_weights: xr.DataArray | None,
) -> xr.DataArray:
    """Get area weights, scaled to an optional region.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing coordinate bounds.
    dv : xr.DataArray
        Data variable defining the spatial dimensions.
    lat_bounds, lon_bounds : tuple[float, float] | None
        Optional regional bounds.
    custom_weights : xr.DataArray | None
        User-supplied area weights, bypassing weight generation.

    Returns
    -------
    xr.DataArray
        Spatial area weights broadcastable to ``dv``. A scalar one is
        returned when the variable has no spatial dimensions.
    """
    spatial_axes = [axis for axis in ("Y", "X") if _has_axis(dv, axis)]

    if not spatial_axes:
        return xr.DataArray(1.0)

    if custom_weights is not None:
        return custom_weights

    return ds.spatial.get_weights(
        axis=spatial_axes,  # type: ignore[arg-type]
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        data_var=str(dv.name),
    )


def _has_axis(dv: xr.DataArray, axis: str) -> bool:
    """Return whether a data variable contains a CF axis.

    Parameters
    ----------
    dv : xr.DataArray
        The data variable.
    axis : str
        CF axis key (e.g. ``"T"``, ``"Y"``).

    Returns
    -------
    bool
        True if the axis dimension is present.
    """
    try:
        get_dim_keys(dv, axis)  # type: ignore[arg-type]
    except KeyError:
        return False
    return True


def _scalar(da: xr.DataArray):
    """Return the bare Python/cftime scalar of a 0-dimensional DataArray.

    ``DataArray.values`` on a 0-d object array yields a 0-d numpy ndarray
    rather than the underlying object. This normalizes both cases so group
    labels are hashable scalars.

    Parameters
    ----------
    da : xr.DataArray
        A 0-dimensional DataArray.

    Returns
    -------
    object
        The scalar value (e.g. a cftime datetime or numpy datetime64).
    """
    value = da.values
    return value.item() if isinstance(value, np.ndarray) else value
