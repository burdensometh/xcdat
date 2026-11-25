"""Tests for the heat exposure statistics module (``xcdat.heat``)."""

import cftime
import numpy as np
import pytest
import xarray as xr

from xcdat.heat import (
    HEAT_FREQUENCIES,
    KELVIN_OFFSET,
    HeatAccessor,
    HeatExposureStats,
)


def build_dataset(
    values: np.ndarray,
    time_bounds: list,
    lat_values: tuple[float, ...] = (-45.0, 45.0),
    lat_bounds: tuple[tuple[float, float], ...] = (
        (-90.0, 0.0),
        (0.0, 90.0),
    ),
    lon_values: tuple[float, ...] = (0.0, 90.0),
    lon_bounds: tuple[tuple[float, float], ...] = (
        (-45.0, 45.0),
        (45.0, 135.0),
    ),
    units: str = "degC",
    calendar: str = "noleap",
) -> xr.Dataset:
    """Build a small, fully-specified dataset for heat exposure tests.

    Parameters
    ----------
    values : np.ndarray
        Temperature values with shape ``(time, lat, lon)``; NaN marks missing.
    time_bounds : list
        Pairs of ``(lower, upper)`` cftime datetimes per time step.
    lat_values, lon_values : tuple
        Coordinate center values.
    lat_bounds, lon_bounds : tuple
        Coordinate bounds per cell.
    units : str
        Temperature units attribute.
    calendar : str
        cftime calendar name.

    Returns
    -------
    xr.Dataset
        Dataset with CF axis attributes and linked bounds.
    """
    date_type = cftime.date2type(calendar) if hasattr(cftime, "date2type") else None
    if date_type is None:
        calendar_map = {
            "noleap": cftime.DatetimeNoLeap,
            "standard": cftime.DatetimeGregorian,
            "360_day": cftime.Datetime360Day,
        }
        date_type = calendar_map[calendar]

    centers = [
        date_type(lower.year, lower.month, min(15, lower.day))
        for lower, _ in time_bounds
    ]
    time = xr.DataArray(
        np.array(centers, dtype=object),
        dims=["time"],
        attrs={"axis": "T", "standard_name": "time"},
    )
    lat = xr.DataArray(
        np.array(lat_values, dtype=float),
        dims=["lat"],
        attrs={
            "axis": "Y",
            "units": "degrees_north",
            "standard_name": "latitude",
        },
    )
    lon = xr.DataArray(
        np.array(lon_values, dtype=float),
        dims=["lon"],
        attrs={
            "axis": "X",
            "units": "degrees_east",
            "standard_name": "longitude",
        },
    )
    tas = xr.DataArray(
        np.array(values, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": units},
        name="tas",
    )
    ds = xr.Dataset({"tas": tas})
    ds["lat_bnds"] = xr.DataArray(
        np.array(lat_bounds, dtype=float), dims=["lat", "bnds"]
    )
    ds["lon_bnds"] = xr.DataArray(
        np.array(lon_bounds, dtype=float), dims=["lon", "bnds"]
    )
    ds["time_bnds"] = xr.DataArray(
        np.array(time_bounds, dtype=object), dims=["time", "bnds"]
    )
    ds.lat.attrs["bounds"] = "lat_bnds"
    ds.lon.attrs["bounds"] = "lon_bnds"
    ds.time.attrs["bounds"] = "time_bnds"
    return ds


@pytest.fixture
def monthly_ds() -> xr.Dataset:
    """Two steps where the first crosses a January/February boundary."""
    time_bounds = [
        (
            cftime.DatetimeNoLeap(2000, 1, 16),
            cftime.DatetimeNoLeap(2000, 2, 15),
        ),
        (
            cftime.DatetimeNoLeap(2000, 2, 15),
            cftime.DatetimeNoLeap(2000, 3, 16),
        ),
    ]
    # NaN at time 0, lat 1, lon 1.
    values = np.array(
        [
            [[35.0, 20.0], [25.0, np.nan]],
            [[40.0, 10.0], [28.0, 5.0]],
        ]
    )
    return build_dataset(values, time_bounds)


class TestHeatAccessor:
    def test_decorator_registration(self, monthly_ds):
        assert isinstance(monthly_ds.heat, HeatAccessor)

    def test_raises_on_missing_variable(self, monthly_ds):
        with pytest.raises(KeyError):
            monthly_ds.heat.exposure("nope", threshold=30.0)

    def test_raises_on_bad_freq(self, monthly_ds):
        with pytest.raises(ValueError):
            monthly_ds.heat.exposure("tas", threshold=30.0, freq="hour")

    def test_raises_on_bad_threshold_units(self, monthly_ds):
        with pytest.raises(ValueError):
            monthly_ds.heat.exposure(
                "tas", threshold=30.0, threshold_units="Rankine"
            )

    def test_raises_when_variable_has_no_units(self, monthly_ds):
        del monthly_ds["tas"].attrs["units"]
        with pytest.raises(ValueError):
            monthly_ds.heat.exposure("tas", threshold=30.0)

    def test_raises_on_unrecognized_temperature_units(self, monthly_ds):
        monthly_ds["tas"].attrs["units"] = "parsec"
        with pytest.raises(ValueError):
            monthly_ds.heat.exposure("tas", threshold=30.0)

    def test_supported_frequencies(self):
        assert set(HEAT_FREQUENCIES) == {"year", "month"}


class TestExposureCalculation:
    def test_returns_stats_dataclass(self, monthly_ds):
        stats = monthly_ds.heat.exposure("tas", threshold=30.0)
        assert isinstance(stats, HeatExposureStats)
        assert stats.threshold == 30.0
        assert stats.data_var == "tas"
        assert stats.freq is None

    def test_exposed_bounded_by_coverage(self, monthly_ds):
        stats = monthly_ds.heat.exposure("tas", threshold=30.0, freq="month")
        assert np.all(stats.exposed.values <= stats.coverage.values + 1e-9)

    def test_missing_observation_excluded_from_both(self, monthly_ds):
        stats = monthly_ds.heat.exposure("tas", threshold=30.0, freq="month")
        # NaN at time 0, whose only January fragment lands in group 0.
        assert stats.coverage.values[0, 1, 1] == 0.0
        assert stats.exposed.values[0, 1, 1] == 0.0
        assert np.isnan(stats.fraction.values[0, 1, 1])

    def test_all_missing_region_is_unknown(self, monthly_ds):
        monthly_ds["tas"].data[:] = np.nan
        stats = monthly_ds.heat.exposure("tas", threshold=30.0)
        assert np.all(stats.coverage.values == 0.0)
        assert np.all(np.isnan(stats.fraction.values))

        grouped = monthly_ds.heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        assert np.all(grouped.coverage.values == 0.0)
        assert np.all(np.isnan(grouped.fraction.values))

    def test_fraction_is_ratio_of_sufficient_statistics(self, monthly_ds):
        stats = monthly_ds.heat.exposure("tas", threshold=30.0)
        expected = stats.exposed.values / stats.coverage.values
        np.testing.assert_allclose(stats.fraction.values, expected)

    def test_ungrouped_equals_sum_of_grouped(self, monthly_ds):
        grouped = monthly_ds.heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        ungrouped = monthly_ds.heat.exposure("tas", threshold=30.0)
        np.testing.assert_allclose(
            grouped.exposed.sum(dim="time").values, ungrouped.exposed.values
        )
        np.testing.assert_allclose(
            grouped.coverage.sum(dim="time").values,
            ungrouped.coverage.values,
        )


class TestMonthBoundarySplit:
    def test_fragments_split_by_overlap_duration(self, monthly_ds):
        stats = monthly_ds.heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        # Three calendar groups: Jan, Feb, Mar.
        assert len(stats.exposed.time) == 3

        area_weights = monthly_ds.spatial.get_weights(
            axis=["Y", "X"], data_var="tas"
        ).values

        # Step 0 spans Jan 16 -> Feb 15 (30 days); 16 days in January.
        jan_seconds = (
            cftime.DatetimeNoLeap(2000, 2, 1)
            - cftime.DatetimeNoLeap(2000, 1, 16)
        ).total_seconds()
        # Cell (lat 0, lon 0) is 35 degC and valid -> fully exposed in Jan.
        np.testing.assert_allclose(
            stats.exposed.values[0, 0, 0],
            jan_seconds * area_weights[0, 0],
        )
        np.testing.assert_allclose(
            stats.coverage.values[0, 0, 0],
            jan_seconds * area_weights[0, 0],
        )
        assert stats.fraction.values[0, 0, 0] == 1.0

    def test_split_step_partial_fraction(self, monthly_ds):
        stats = monthly_ds.heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        # February group: step 0 tail (35 degC, 14 days) + step 1 head
        # (40 degC, 14 days) for cell (0, 0); both exposed -> fraction 1.
        np.testing.assert_allclose(stats.fraction.values[1, 0, 0], 1.0)
        # Cell (0, 1): step 0 = 20 (not exposed), step 1 = 10 (not exposed).
        assert stats.fraction.values[1, 0, 1] == 0.0

    def test_fragment_weights_preserve_total_duration(self, monthly_ds):
        from xcdat.heat import _build_time_fragments

        weights, _, source = _build_time_fragments(
            monthly_ds,
            monthly_ds["tas"],
            "month",
            "time_fragment",
        )
        time_bnds = monthly_ds.time_bnds
        total = sum(
            (time_bnds[i, 1].values - time_bnds[i, 0].values).total_seconds()
            for i in range(time_bnds.sizes["time"])
        )
        np.testing.assert_allclose(float(weights.sum()), total)
        # First step is split -> two fragments referencing source index 0.
        assert list(source.values) == [0, 0, 1, 1]

    def test_zero_duration_tail_on_boundary_is_omitted(self):
        # A step whose upper bound lands exactly on a month boundary must not
        # emit an empty zero-weight fragment for the next month.
        time_bounds = [
            (
                cftime.DatetimeNoLeap(2000, 1, 1),
                cftime.DatetimeNoLeap(2000, 2, 1),
            )
        ]
        values = np.array([[[35.0, 10.0]]])
        ds = build_dataset(
            values,
            time_bounds,
            lat_values=(0.0,),
            lat_bounds=((-1.0, 1.0),),
            lon_values=(0.0, 90.0),
            lon_bounds=((-45.0, 45.0), (45.0, 135.0)),
        )
        stats = ds.heat.exposure("tas", threshold=30.0, freq="month")
        assert len(stats.exposed.time) == 1
        np.testing.assert_allclose(
            stats.fraction.values[0], [[1.0, 0.0]]
        )


class TestYearBoundarySplit:
    def test_year_split_across_boundary(self):
        time_bounds = [
            (
                cftime.DatetimeNoLeap(1999, 12, 16),
                cftime.DatetimeNoLeap(2000, 1, 15),
            ),
            (
                cftime.DatetimeNoLeap(2000, 1, 15),
                cftime.DatetimeNoLeap(2000, 2, 14),
            ),
        ]
        values = np.array([[[35.0]], [[10.0]]])
        ds = build_dataset(
            values,
            time_bounds,
            lat_values=(0.0,),
            lat_bounds=((-1.0, 1.0),),
            lon_values=(0.0,),
            lon_bounds=((-1.0, 1.0),),
        )
        stats = ds.heat.exposure("tas", threshold=30.0, freq="year")
        assert len(stats.exposed.time) == 2
        # 1999 fragment holds only the exposed step-0 head.
        assert stats.fraction.values[0, 0, 0] == 1.0
        # 2000: 14 exposed days (step 0 tail) over 14 + 30 total days.
        np.testing.assert_allclose(
            stats.fraction.values[1, 0, 0], 14.0 / (14.0 + 30.0)
        )


class TestRegionAreaWeighting:
    def test_outside_region_zero_weight(self, monthly_ds):
        stats = monthly_ds.heat.exposure(
            "tas", threshold=30.0, lat_bounds=(-90.0, 0.0)
        )
        # Northern row (lat index 1) is outside the region.
        assert np.all(stats.coverage.values[1, :] == 0.0)
        assert np.all(np.isnan(stats.fraction.values[1, :]))

    def test_partial_cell_scaled_by_overlap_area(self, monthly_ds):
        full = monthly_ds.heat.exposure("tas", threshold=30.0)
        partial = monthly_ds.heat.exposure(
            "tas", threshold=30.0, lat_bounds=(-45.0, 45.0)
        )
        scale = (
            np.sin(np.radians(45.0)) - np.sin(0.0)
        )  # clipped sine span per cell
        # Full cells have sine span sin(90) - sin(0) = 1.
        np.testing.assert_allclose(
            partial.coverage.values, full.coverage.values * scale
        )

    def test_custom_weights_used(self, monthly_ds):
        custom = xr.DataArray(
            np.full((2, 2), 2.0),
            dims=["lat", "lon"],
            coords={"lat": monthly_ds.lat, "lon": monthly_ds.lon},
        )
        default = monthly_ds.heat.exposure("tas", threshold=30.0)
        weighted = monthly_ds.heat.exposure(
            "tas", threshold=30.0, weights=custom
        )
        area = monthly_ds.spatial.get_weights(
            axis=["Y", "X"], data_var="tas"
        )
        # Custom weight 2 replaces the generated area weight (90 per cell).
        np.testing.assert_allclose(
            weighted.coverage.values,
            default.coverage.values * (2.0 / area.values),
        )


class TestUnitConversion:
    def test_celsius_and_kelvin_inputs_agree(self, monthly_ds):
        celsius = monthly_ds.heat.exposure(
            "tas", threshold=30.0, threshold_units="degC", freq="month"
        )

        kelvin_ds = monthly_ds.copy(deep=True)
        kelvin_ds["tas"] = kelvin_ds["tas"] + KELVIN_OFFSET
        kelvin_ds["tas"].attrs["units"] = "K"
        kelvin = kelvin_ds.heat.exposure(
            "tas",
            threshold=30.0 + KELVIN_OFFSET,
            threshold_units="K",
            freq="month",
        )

        np.testing.assert_allclose(
            kelvin.exposed.values, celsius.exposed.values
        )
        np.testing.assert_allclose(
            kelvin.coverage.values, celsius.coverage.values
        )
        np.testing.assert_allclose(
            kelvin.fraction.values,
            celsius.fraction.values,
            equal_nan=True,
        )

    def test_kelvin_threshold_on_celsius_data(self, monthly_ds):
        celsius = monthly_ds.heat.exposure("tas", threshold=30.0)
        kelvin_threshold = monthly_ds.heat.exposure(
            "tas", threshold=30.0 + KELVIN_OFFSET, threshold_units="K"
        )
        np.testing.assert_allclose(
            kelvin_threshold.fraction.values,
            celsius.fraction.values,
            equal_nan=True,
        )

    def test_celsius_threshold_on_kelvin_data(self, monthly_ds):
        kelvin_ds = monthly_ds.copy(deep=True)
        kelvin_ds["tas"] = kelvin_ds["tas"] + KELVIN_OFFSET
        kelvin_ds["tas"].attrs["units"] = "K"
        from_celsius = kelvin_ds.heat.exposure(
            "tas", threshold=30.0, threshold_units="degC"
        )
        native = kelvin_ds.heat.exposure(
            "tas",
            threshold=30.0 + KELVIN_OFFSET,
            threshold_units="K",
        )
        np.testing.assert_allclose(
            from_celsius.fraction.values,
            native.fraction.values,
            equal_nan=True,
        )


class TestMerge:
    def test_merge_adjacent_time_periods_matches_direct(self, monthly_ds):
        direct = monthly_ds.heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        first = monthly_ds.isel(time=[0]).heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        second = monthly_ds.isel(time=[1]).heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        merged = first.merge(second)

        np.testing.assert_allclose(
            merged.exposed.values, direct.exposed.values
        )
        np.testing.assert_allclose(
            merged.coverage.values, direct.coverage.values
        )
        np.testing.assert_allclose(
            merged.fraction.values,
            direct.fraction.values,
            equal_nan=True,
        )

    def test_merge_disjoint_regions_matches_direct(self, monthly_ds):
        direct = monthly_ds.heat.exposure("tas", threshold=30.0)
        left = monthly_ds.isel(lon=[0]).heat.exposure(
            "tas", threshold=30.0
        )
        right = monthly_ds.isel(lon=[1]).heat.exposure(
            "tas", threshold=30.0
        )
        merged = left.merge(right)

        np.testing.assert_allclose(
            merged.exposed.values, direct.exposed.values
        )
        np.testing.assert_allclose(
            merged.coverage.values, direct.coverage.values
        )
        np.testing.assert_allclose(
            merged.fraction.values,
            direct.fraction.values,
            equal_nan=True,
        )

    def test_merge_is_associative(self, monthly_ds):
        a = monthly_ds.isel(time=[0]).heat.exposure(
            "tas", threshold=30.0
        )
        b = monthly_ds.isel(time=[1], lon=[0]).heat.exposure(
            "tas", threshold=30.0
        )
        c = monthly_ds.isel(time=[1], lon=[1]).heat.exposure(
            "tas", threshold=30.0
        )
        direct = monthly_ds.heat.exposure("tas", threshold=30.0)

        left_assoc = a.merge(b).merge(c)
        right_assoc = a.merge(b.merge(c))
        np.testing.assert_allclose(
            left_assoc.coverage.values, right_assoc.coverage.values
        )
        np.testing.assert_allclose(
            left_assoc.exposed.values, direct.exposed.values
        )

    def test_merge_rejects_mismatched_threshold(self, monthly_ds):
        a = monthly_ds.heat.exposure("tas", threshold=30.0)
        b = monthly_ds.heat.exposure("tas", threshold=31.0)
        with pytest.raises(ValueError):
            a.merge(b)

    def test_merge_rejects_mismatched_units(self, monthly_ds):
        a = monthly_ds.heat.exposure(
            "tas", threshold=30.0, threshold_units="degC"
        )
        b = monthly_ds.heat.exposure(
            "tas", threshold=30.0, threshold_units="K"
        )
        with pytest.raises(ValueError):
            a.merge(b)

    def test_merge_rejects_mismatched_freq(self, monthly_ds):
        a = monthly_ds.heat.exposure("tas", threshold=30.0)
        b = monthly_ds.heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        with pytest.raises(ValueError):
            a.merge(b)

    def test_merge_rejects_mismatched_variable(self, monthly_ds):
        other = monthly_ds.rename({"tas": "tas2"})
        a = monthly_ds.heat.exposure("tas", threshold=30.0)
        b = other.heat.exposure("tas2", threshold=30.0)
        with pytest.raises(ValueError):
            a.merge(b)


class TestNumpyDatetimeCalendar:
    def test_datetime64_monthly_and_yearly(self):
        time_bounds = np.array(
            [
                ["1999-12-16T00:00", "2000-01-15T00:00"],
                ["2000-01-15T00:00", "2000-02-14T00:00"],
            ],
            dtype="datetime64[ns]",
        )
        time = xr.DataArray(
            np.array(["1999-12-31", "2000-01-30"], dtype="datetime64[ns]"),
            dims=["time"],
            attrs={"axis": "T", "standard_name": "time"},
        )
        lat = xr.DataArray(
            [0.0],
            dims=["lat"],
            attrs={
                "axis": "Y",
                "units": "degrees_north",
                "standard_name": "latitude",
            },
        )
        lon = xr.DataArray(
            [0.0],
            dims=["lon"],
            attrs={
                "axis": "X",
                "units": "degrees_east",
                "standard_name": "longitude",
            },
        )
        tas = xr.DataArray(
            np.array([[[35.0]], [[10.0]]]),
            dims=["time", "lat", "lon"],
            coords={"time": time, "lat": lat, "lon": lon},
            attrs={"units": "degC"},
            name="tas",
        )
        ds = xr.Dataset({"tas": tas})
        ds["lat_bnds"] = xr.DataArray(
            [[-1.0, 1.0]], dims=["lat", "bnds"]
        )
        ds["lon_bnds"] = xr.DataArray(
            [[-1.0, 1.0]], dims=["lon", "bnds"]
        )
        ds["time_bnds"] = xr.DataArray(
            time_bounds, dims=["time", "bnds"]
        )
        ds.lat.attrs["bounds"] = "lat_bnds"
        ds.lon.attrs["bounds"] = "lon_bnds"
        ds.time.attrs["bounds"] = "time_bnds"

        yearly = ds.heat.exposure("tas", threshold=30.0, freq="year")
        assert np.issubdtype(yearly.exposed.time.dtype, np.datetime64)
        np.testing.assert_allclose(
            yearly.fraction.values[1, 0, 0], 14.0 / (14.0 + 30.0)
        )

        monthly = ds.heat.exposure("tas", threshold=30.0, freq="month")
        assert len(monthly.exposed.time) == 3


class TestNoSpatialDims:
    def test_time_only_variable(self, monthly_ds):
        ds1d = monthly_ds.isel(lat=[0], lon=[0]).squeeze(["lat", "lon"])
        stats = ds1d.heat.exposure("tas", threshold=30.0)
        assert stats.exposed.dims == ()
        assert np.isfinite(stats.fraction.values)


class TestDaskSupport:
    def test_chunked_data_matches_eager(self, monthly_ds):
        dask_ds = monthly_ds.chunk({"time": 1})
        eager = monthly_ds.heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        chunked = dask_ds.heat.exposure(
            "tas", threshold=30.0, freq="month"
        )
        np.testing.assert_allclose(
            chunked.exposed.values, eager.exposed.values
        )
        np.testing.assert_allclose(
            chunked.coverage.values, eager.coverage.values
        )
        np.testing.assert_allclose(
            chunked.fraction.values,
            eager.fraction.values,
            equal_nan=True,
        )


class TestTimedeltaConversion:
    def test_object_dtype_timedelta(self):
        import datetime

        from xcdat.heat import _timedelta_to_float

        delta = xr.DataArray(
            np.array(
                [datetime.timedelta(seconds=10), datetime.timedelta(seconds=20)]
            ),
            dims=["t"],
        )
        result = _timedelta_to_float(delta)
        np.testing.assert_allclose(result.values, [10.0, 20.0])


class TestToDataset:
    def test_dataset_contains_all_variables(self, monthly_ds):
        stats = monthly_ds.heat.exposure("tas", threshold=30.0, freq="month")
        ds_out = stats.to_dataset()
        assert "tas_heat_exposed" in ds_out.data_vars
        assert "tas_heat_coverage" in ds_out.data_vars
        assert "tas_heat_exposure_fraction" in ds_out.data_vars
        assert ds_out.attrs["heat_exposure_threshold"] == 30.0
        assert ds_out.attrs["heat_exposure_freq"] == "month"
