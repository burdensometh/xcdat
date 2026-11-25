"""Tests for the circular (directional) statistics module."""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import requires_dask

import xcdat  # noqa: F401
from xcdat.circular import (
    CircularAccessor,
    _derive_direction_and_concentration,
    _overlap_matrix,
)


def _make_dataset(
    times,
    bounds,
    values,
    extra_dims: dict[str, int] | None = None,
    dtype: str | None = "datetime64[ns]",
) -> xr.Dataset:
    """Build a Dataset with a direction variable and time bounds.

    Parameters
    ----------
    times : list
        The time coordinates. Values are cast to ``dtype`` for NumPy datetimes,
        or kept as objects for ``cftime`` datetimes.
    bounds : list[list]
        The time bounds for each time coordinate.
    values : np.ndarray
        The direction values. Shape must be ``(len(times), ...)``.
    extra_dims : dict[str, int] | None
        Additional dimensions after ``time`` with their sizes.
    dtype : str | None
        The NumPy datetime dtype. Set to ``None`` to keep the values as
        ``cftime`` datetime objects.

    Returns
    -------
    xr.Dataset
        The Dataset.
    """
    values = np.asarray(values, dtype="float64")

    if dtype is None:
        time_values = np.array(times, dtype=object)
        bounds_values = np.array(bounds, dtype=object)
    else:
        time_values = np.array(times, dtype=dtype)
        bounds_values = np.array(bounds, dtype=dtype)

    time = xr.DataArray(
        time_values,
        dims=["time"],
        attrs={
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": "time_bnds",
        },
    )
    time_bnds = xr.DataArray(
        bounds_values,
        dims=["time", "bnds"],
        attrs={"xcdat_bounds": "True"},
    )

    dims = ("time",) if extra_dims is None else ("time", *extra_dims.keys())
    coords: dict[str, object] = {"time": time, "time_bnds": time_bnds}
    if extra_dims is not None:
        for name, size in extra_dims.items():
            coords[name] = np.arange(size)

    return xr.Dataset({"wd": (dims, values)}, coords=coords)


def _hourly_dataset(
    start: str = "2000-01-01", periods: int = 24, values=None, extra_dims=None
) -> xr.Dataset:
    """Build an hourly Dataset with one-hour observation bounds."""
    times = pd.date_range(start, periods=periods, freq="h")
    bounds = np.stack(
        [times.values, (times + pd.Timedelta(hours=1)).values], axis=1
    )
    if values is None:
        values = (np.arange(periods) * 15.0) % 360.0

    return _make_dataset(times.values, bounds, values, extra_dims=extra_dims)


class TestCircularAccessor:
    def test__init__(self):
        ds = _hourly_dataset()
        obj = CircularAccessor(ds)
        assert obj._dataset.identical(ds)

    def test_decorator(self):
        ds = _hourly_dataset()
        assert ds.circular._dataset.identical(ds)


class TestSummary:
    def test_duration_weighted_direction(self):
        # Two hours from 90° and three hours from 0° are not equal in duration.
        times = ["2000-01-01T00:00", "2000-01-01T01:00", "2000-01-01T02:00"]
        bounds = [
            ["2000-01-01T00:00", "2000-01-01T01:00"],
            ["2000-01-01T01:00", "2000-01-01T02:00"],
            ["2000-01-01T02:00", "2000-01-01T05:00"],
        ]
        ds = _make_dataset(times, bounds, [90.0, 90.0, 0.0])

        result = ds.circular.summary("wd", freq="day")

        # Work in hours; the direction ratio is unit-independent.
        east = 2.0
        north = 3.0
        expected_direction = np.degrees(np.arctan2(east, north)) % 360.0
        expected_concentration = np.hypot(east, north) / 5.0

        assert result["wd_direction"].values == pytest.approx(
            [expected_direction], abs=1e-9
        )
        assert result["wd_weight"].values == pytest.approx([5 * 3600.0])
        assert result["wd_concentration"].values == pytest.approx(
            [expected_concentration]
        )

    def test_observation_split_across_days_by_actual_duration(self):
        # A single six-hour observation from 21:00 to 03:00 spans two days and
        # must contribute three hours to each daily report.
        ds = _make_dataset(
            ["2000-01-01T21:00"],
            [["2000-01-01T21:00", "2000-01-02T03:00"]],
            [0.0],
        )

        result = ds.circular.summary("wd", freq="day")

        np.testing.assert_allclose(result["wd_weight"].values, [3 * 3600, 3 * 3600])
        np.testing.assert_allclose(result["wd_direction"].values, [0.0, 0.0])
        np.testing.assert_allclose(
            result["wd_resultant_east"].values, [0.0, 0.0], atol=1e-9
        )
        np.testing.assert_allclose(
            result["wd_resultant_north"].values, [3 * 3600, 3 * 3600]
        )

    def test_missing_directions_are_excluded(self):
        times = ["2000-01-01T00:00", "2000-01-01T06:00", "2000-01-01T12:00"]
        bounds = [
            ["2000-01-01T00:00", "2000-01-01T06:00"],
            ["2000-01-01T06:00", "2000-01-01T12:00"],
            ["2000-01-01T12:00", "2000-01-01T18:00"],
        ]
        ds = _make_dataset(times, bounds, [90.0, np.nan, 90.0])

        result = ds.circular.summary("wd", freq="day")

        assert result["wd_direction"].values[0] == pytest.approx(90.0)
        assert result["wd_concentration"].values[0] == pytest.approx(1.0)
        # The missing observation contributes no duration.
        assert result["wd_weight"].values[0] == pytest.approx(12 * 3600.0)

    def test_true_north_is_preserved(self):
        ds = _make_dataset(
            ["2000-01-01T00:00"],
            [["2000-01-01T00:00", "2000-01-02T00:00"]],
            [0.0],
        )

        result = ds.circular.summary("wd", freq="day")

        assert result["wd_direction"].values[0] == 0.0
        assert not np.isnan(result["wd_direction"].values[0])
        assert result["wd_concentration"].values[0] == pytest.approx(1.0)

    def test_full_turn_is_normalized_to_true_north(self):
        ds = _make_dataset(
            ["2000-01-01T00:00"],
            [["2000-01-01T00:00", "2000-01-02T00:00"]],
            [360.0],
        )

        result = ds.circular.summary("wd", freq="day")

        assert result["wd_direction"].values[0] == 0.0

    def test_direction_wraps_across_zero(self):
        # 350° and 10° average across the discontinuity at north.
        ds = _make_dataset(
            ["2000-01-01T00:00", "2000-01-01T01:00"],
            [
                ["2000-01-01T00:00", "2000-01-01T01:00"],
                ["2000-01-01T01:00", "2000-01-01T02:00"],
            ],
            [350.0, 10.0],
        )

        result = ds.circular.summary("wd", freq="day")

        assert result["wd_direction"].values[0] == pytest.approx(0.0, abs=1e-9)

    def test_all_missing_direction_is_undefined(self):
        ds = _make_dataset(
            ["2000-01-01T00:00"],
            [["2000-01-01T00:00", "2000-01-02T00:00"]],
            [np.nan],
        )

        result = ds.circular.summary("wd", freq="day")

        assert np.isnan(result["wd_direction"].values[0])
        assert np.isnan(result["wd_concentration"].values[0])
        assert result["wd_weight"].values[0] == 0.0

    def test_opposing_equal_winds_are_indeterminate(self):
        ds = _make_dataset(
            ["2000-01-01T00:00", "2000-01-01T12:00"],
            [
                ["2000-01-01T00:00", "2000-01-01T12:00"],
                ["2000-01-01T12:00", "2000-01-02T00:00"],
            ],
            [90.0, 270.0],
        )

        result = ds.circular.summary("wd", freq="day")

        assert np.isnan(result["wd_direction"].values[0])
        assert result["wd_concentration"].values[0] == 0.0

    def test_opposing_unequal_winds_have_a_direction(self):
        # A 12-hour wind from 90° against a 36-hour wind from 270° leaves a
        # non-zero resultant pointing at 270° with a concentration of 0.5.
        ds = _make_dataset(
            ["2000-01-01T00:00", "2000-01-01T12:00"],
            [
                ["2000-01-01T00:00", "2000-01-01T12:00"],
                ["2000-01-01T12:00", "2000-01-03T00:00"],
            ],
            [90.0, 270.0],
        )

        result = ds.circular.summary("wd", freq="month")

        assert result["wd_direction"].values[0] == pytest.approx(270.0)
        assert result["wd_concentration"].values[0] == pytest.approx(0.5)

    def test_min_concentration_threshold_sets_indeterminate(self):
        # Two nearly opposing winds with a tiny resultant.
        ds = _make_dataset(
            ["2000-01-01T00:00", "2000-01-01T12:00"],
            [
                ["2000-01-01T00:00", "2000-01-01T12:00"],
                ["2000-01-01T12:00", "2000-01-02T00:00"],
            ],
            [0.0, 179.0],
        )

        result = ds.circular.summary(
            "wd", freq="day", min_concentration=0.01
        )

        assert np.isnan(result["wd_direction"].values[0])
        assert result["wd_concentration"].values[0] == 0.0

    def test_opposing_equal_winds_are_indeterminate_for_float32_input(self):
        # float32 input must not leave a spurious residual vector that would
        # appear as a well-defined direction.
        ds = _make_dataset(
            ["2000-01-01T00:00", "2000-01-02T00:00"],
            [
                ["2000-01-01T00:00", "2000-01-02T00:00"],
                ["2000-01-02T00:00", "2000-01-03T00:00"],
            ],
            np.array([90.0, 270.0], dtype="float32"),
        )

        result = ds.circular.summary("wd", freq="month")

        assert np.isnan(result["wd_direction"].values[0])
        assert result["wd_concentration"].values[0] == 0.0

    def test_period_labels_and_bounds(self):
        ds = _hourly_dataset(periods=48)

        result = ds.circular.summary("wd", freq="day")

        expected = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(result["time"].values, expected)
        assert result["time"].attrs["bounds"] == "time_bnds"
        assert result["time_bnds"].attrs["xcdat_bounds"] == "True"
        np.testing.assert_array_equal(
            result["time_bnds"].values,
            np.array(
                [
                    ["2000-01-01T00:00", "2000-01-02T00:00"],
                    ["2000-01-02T00:00", "2000-01-03T00:00"],
                ],
                dtype="datetime64[ns]",
            ),
        )

    def test_report_variable_metadata(self):
        ds = _hourly_dataset(periods=24, values=np.full(24, 45.0))

        result = ds.circular.summary("wd", freq="day")

        for suffix in (
            "direction",
            "concentration",
            "resultant_east",
            "resultant_north",
            "weight",
        ):
            attrs = result[f"wd_{suffix}"].attrs
            assert attrs["circular_report"] == "True"
            assert attrs["circular_freq"] == "day"
            assert attrs["circular_source_var"] == "wd"
            assert attrs["operation"] == "circular_summary"
        assert result["wd_direction"].attrs["units"] == "degrees"
        assert result["wd_weight"].attrs["units"] == "s"

    def test_multidimensional_data_variable(self):
        values = np.array([[45.0, 90.0], [45.0, 180.0], [45.0, 180.0], [45.0, 0.0]])
        ds = _make_dataset(
            [
                "2000-01-01T00:00",
                "2000-01-01T01:00",
                "2000-01-01T02:00",
                "2000-01-01T03:00",
            ],
            [
                ["2000-01-01T00:00", "2000-01-01T01:00"],
                ["2000-01-01T01:00", "2000-01-01T02:00"],
                ["2000-01-01T02:00", "2000-01-01T03:00"],
                ["2000-01-01T03:00", "2000-01-01T04:00"],
            ],
            values,
            extra_dims={"station": 2},
        )

        result = ds.circular.summary("wd", freq="day")

        assert result["wd_direction"].dims == ("time", "station")
        np.testing.assert_allclose(
            result["wd_direction"].values, [[45.0, 135.0]], atol=1e-9
        )
        np.testing.assert_allclose(
            result["wd_concentration"].values,
            [[1.0, np.hypot(1.0, 1.0) / 4.0]],
            atol=1e-9,
        )

    def test_non_time_coordinate_attributes_are_preserved(self):
        rng = np.random.default_rng(1)
        values = rng.uniform(0, 360, size=(48, 2))
        ds = _hourly_dataset(periods=48, values=values, extra_dims={"lat": 2})
        ds = ds.assign_coords(
            lat=xr.DataArray(
                [0.0, 1.0],
                dims="lat",
                attrs={
                    "axis": "Y",
                    "units": "degrees_north",
                    "standard_name": "latitude",
                },
            )
        )

        result = ds.circular.summary("wd", freq="day")
        combined = result.circular.combine("wd", freq="month")

        for report in (result, combined):
            assert report["lat"].attrs["units"] == "degrees_north"
            assert report["lat"].attrs["standard_name"] == "latitude"
            assert report["wd_direction"].attrs["units"] == "degrees"

    @requires_dask
    def test_dask_backed_data_variable_is_supported(self):
        ds = _hourly_dataset(periods=48).chunk({"time": 12})

        result = ds.circular.summary("wd", freq="day")

        assert result["wd_direction"].chunks is not None
        eager = ds.compute().circular.summary("wd", freq="day")
        np.testing.assert_allclose(
            result["wd_direction"].compute().values, eager["wd_direction"].values
        )

    def test_cftime_datetimes_with_different_calendars(self):
        calendars = [
            ("noleap", cftime.DatetimeNoLeap),
            ("360_day", cftime.Datetime360Day),
            ("all_leap", cftime.DatetimeAllLeap),
            ("proleptic_gregorian", cftime.DatetimeProlepticGregorian),
        ]

        for calendar, date_type in calendars:
            times = [
                date_type(2000, 2, 27, 22),
                date_type(2000, 2, 27, 23),
                date_type(2000, 2, 28, 0),
                date_type(2000, 2, 28, 1),
            ]
            upper = date_type(2000, 2, 28, 2)
            bounds = [
                [times[0], times[1]],
                [times[1], times[2]],
                [times[2], times[3]],
                [times[3], upper],
            ]
            ds = _make_dataset(
                times, bounds, [10.0, 10.0, 10.0, 10.0], dtype=None
            )
            ds["time"].encoding["calendar"] = calendar
            ds["time_bnds"].encoding["calendar"] = calendar

            result = ds.circular.summary("wd", freq="day")

            assert result["time"].encoding["calendar"] == calendar
            np.testing.assert_allclose(result["wd_direction"].values, [10.0, 10.0])
            assert isinstance(result["time"].values[0], cftime.datetime)


class TestCombine:
    def test_combine_requires_a_summary_report(self):
        ds = _hourly_dataset(periods=24)

        with pytest.raises(KeyError, match="circular report variables"):
            ds.circular.combine("wd", freq="day")

    def test_raises_error_for_finer_target_frequency(self):
        ds = _hourly_dataset(periods=24)
        monthly = ds.circular.summary("wd", freq="month")

        with pytest.raises(ValueError, match="finer frequency"):
            monthly.circular.combine("wd", freq="day")

    def test_combine_matches_direct_summary_for_daily_to_monthly(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(0, 360, size=(24 * 40,))
        # Include missing values and a full turn.
        values[3] = np.nan
        values[10] = 360.0
        ds = _hourly_dataset(start="2000-01-01", periods=24 * 40, values=values)

        daily = ds.circular.summary("wd", freq="day")
        combined = daily.circular.combine("wd", freq="month")
        direct = ds.circular.summary("wd", freq="month")

        xr.testing.assert_allclose(combined, direct)

    def test_combine_matches_direct_summary_for_hourly_to_day(self):
        rng = np.random.default_rng(7)
        values = rng.uniform(0, 360, size=(24 * 5,))
        ds = _hourly_dataset(periods=24 * 5, values=values)

        hourly = ds.circular.summary("wd", freq="hour")
        combined = hourly.circular.combine("wd", freq="day")
        direct = ds.circular.summary("wd", freq="day")

        xr.testing.assert_allclose(combined, direct)

    def test_combine_matches_direct_summary_for_daily_to_yearly(self):
        rng = np.random.default_rng(11)
        ds = _hourly_dataset(
            start="2000-01-01", periods=24 * 400, values=rng.uniform(0, 360, 24 * 400)
        )

        daily = ds.circular.summary("wd", freq="day")
        combined = daily.circular.combine("wd", freq="year")
        direct = ds.circular.summary("wd", freq="year")

        xr.testing.assert_allclose(combined, direct)

    def test_same_frequency_combine_preserves_values(self):
        rng = np.random.default_rng(3)
        ds = _hourly_dataset(periods=24 * 10, values=rng.uniform(0, 360, 24 * 10))

        daily = ds.circular.summary("wd", freq="day")
        unchanged = daily.circular.combine("wd", freq="day")

        xr.testing.assert_allclose(unchanged, daily)

    def test_opposing_equal_winds_on_different_days_are_indeterminate(self):
        # Two equal-duration winds blowing in exactly opposite directions on
        # separate days must merge to an indeterminate direction.
        times = ["2000-01-01T00:00", "2000-01-02T00:00"]
        bounds = [
            ["2000-01-01T00:00", "2000-01-02T00:00"],
            ["2000-01-02T00:00", "2000-01-03T00:00"],
        ]
        ds = _make_dataset(times, bounds, [90.0, 270.0])

        daily = ds.circular.summary("wd", freq="day")
        assert not np.isnan(daily["wd_direction"].values).any()

        combined = daily.circular.combine("wd", freq="month")
        direct = ds.circular.summary("wd", freq="month")

        assert np.isnan(combined["wd_direction"].values[0])
        assert combined["wd_concentration"].values[0] == 0.0
        xr.testing.assert_allclose(combined, direct)

    @pytest.mark.parametrize(
        "direction_a, direction_b",
        [(0.0, 180.0), (45.0, 225.0), (10.0, 190.0), (120.0, 300.0)],
    )
    def test_exactly_opposing_pair_is_indeterminate(self, direction_a, direction_b):
        times = ["2000-01-01T00:00", "2000-01-05T00:00"]
        bounds = [
            ["2000-01-01T00:00", "2000-01-02T00:00"],
            ["2000-01-05T00:00", "2000-01-06T00:00"],
        ]
        ds = _make_dataset(times, bounds, [direction_a, direction_b])

        result = ds.circular.summary("wd", freq="month")

        assert np.isnan(result["wd_direction"].values[0])
        assert result["wd_concentration"].values[0] == 0.0

    def test_multidimensional_combine_matches_direct(self):
        rng = np.random.default_rng(5)
        values = rng.uniform(0, 360, size=(24 * 40, 2, 3))
        values[0, 0, 0] = np.nan
        ds = _hourly_dataset(
            periods=24 * 40, values=values, extra_dims={"lat": 2, "lon": 3}
        )

        daily = ds.circular.summary("wd", freq="day")
        combined = daily.circular.combine("wd", freq="month")
        direct = ds.circular.summary("wd", freq="month")

        xr.testing.assert_allclose(combined, direct)

    def test_cftime_report_can_be_combined(self):
        times = [
            cftime.DatetimeNoLeap(2000, 1, 31, 22),
            cftime.DatetimeNoLeap(2000, 2, 1, 0),
        ]
        bounds = [
            [times[0], times[1]],
            [times[1], cftime.DatetimeNoLeap(2000, 2, 1, 2)],
        ]
        ds = _make_dataset(times, bounds, [90.0, 90.0], dtype=None)
        ds["time"].encoding["calendar"] = "noleap"
        ds["time_bnds"].encoding["calendar"] = "noleap"

        daily = ds.circular.summary("wd", freq="day")
        combined = daily.circular.combine("wd", freq="month")
        direct = ds.circular.summary("wd", freq="month")

        xr.testing.assert_allclose(combined, direct)

    def test_report_round_trips_through_netcdf(self, tmp_path):
        rng = np.random.default_rng(9)
        values = rng.uniform(0, 360, size=(24 * 40,))
        values[5] = np.nan
        ds = _hourly_dataset(periods=24 * 40, values=values)
        ds["time"].encoding.update(
            {"units": "hours since 2000-01-01", "calendar": "standard"}
        )
        ds["time_bnds"].encoding.update(
            {"units": "hours since 2000-01-01", "calendar": "standard"}
        )

        daily = ds.circular.summary("wd", freq="day")
        path = tmp_path / "daily.nc"
        daily.to_netcdf(path)

        reopened = xr.open_dataset(path, decode_times=True).load()
        combined = reopened.circular.combine("wd", freq="month")
        direct = ds.circular.summary("wd", freq="month")

        xr.testing.assert_allclose(combined, direct)


class TestErrors:
    def test_raises_error_with_unsupported_frequency(self):
        ds = _hourly_dataset(periods=24)

        with pytest.raises(ValueError, match="Incorrect `freq` argument"):
            ds.circular.summary("wd", freq="week")

    def test_raises_error_with_invalid_min_concentration(self):
        ds = _hourly_dataset(periods=24)

        for value in (-0.1, 1.1):
            with pytest.raises(ValueError, match="`min_concentration`"):
                ds.circular.summary("wd", freq="day", min_concentration=value)

    def test_raises_error_if_data_variable_is_missing(self):
        ds = _hourly_dataset(periods=24)

        with pytest.raises(KeyError, match="does not exist"):
            ds.circular.summary("not_a_var", freq="day")

    def test_raises_error_if_data_variable_is_not_numeric(self):
        ds = _hourly_dataset(periods=24)
        ds["wd"] = ds["wd"].astype(str)

        with pytest.raises(TypeError, match="must be numeric"):
            ds.circular.summary("wd", freq="day")

    def test_raises_error_if_time_coords_are_not_decoded(self):
        ds = _hourly_dataset(periods=24)
        ds = ds.assign_coords({"time": ds["time"].astype(str)})

        with pytest.raises(TypeError, match="must be decoded"):
            ds.circular.summary("wd", freq="day")

    def test_raises_error_if_time_bounds_are_missing(self):
        ds = _hourly_dataset(periods=24)
        ds = ds.drop_vars("time_bnds")
        ds["time"].attrs.pop("bounds")

        with pytest.raises(KeyError, match="No bounds data variables"):
            ds.circular.summary("wd", freq="day")

    def test_raises_error_if_time_bounds_are_multidimensional(self):
        ds = _hourly_dataset(periods=24)
        bounds = np.stack([ds["time"].values, ds["time"].values], axis=1)
        ds["time_bnds"] = xr.DataArray(
            np.stack([bounds, bounds], axis=1),
            dims=["time", "extra", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        with pytest.raises(ValueError, match="exactly one more dimension"):
            ds.circular.summary("wd", freq="day")

    def test_raises_error_if_no_observation_has_positive_duration(self):
        ds = _make_dataset(
            ["2000-01-01T00:00"],
            [["2000-01-01T00:00", "2000-01-01T00:00"]],
            [45.0],
        )

        with pytest.raises(ValueError, match="finite, positive duration"):
            ds.circular.summary("wd", freq="day")


class TestHelpers:
    def test_overlap_matrix_single_and_crossing_periods(self):
        lower = np.array([0.0, 3 * 86400 + 3600, 86400 + 1])
        upper = np.array([3600.0, 4 * 86400, 86400 + 2])
        edges = np.array([0.0, 86400.0, 2 * 86400.0, 3 * 86400.0, 4 * 86400.0])

        overlap = _overlap_matrix(lower, upper, edges)

        np.testing.assert_allclose(overlap.sum(axis=0), upper - lower)
        assert overlap[0, 0] == pytest.approx(3600.0)
        assert overlap[3, 1] == pytest.approx(82800.0)
        assert overlap[1, 2] == pytest.approx(1.0)

    def test_overlap_matrix_ignores_invalid_observations(self):
        lower = np.array([0.0, np.nan])
        upper = np.array([3600.0, 7200.0])
        edges = np.array([0.0, 86400.0])

        overlap = _overlap_matrix(lower, upper, edges)

        np.testing.assert_allclose(overlap, [[3600.0, 0.0]])

    def test_derive_direction_indeterminate_and_undefined(self):
        east = xr.DataArray([0.0, 0.0, 1.0])
        north = xr.DataArray([0.0, 1.0, 0.0])
        weight = xr.DataArray([0.0, 1.0, 1.0])

        direction, concentration = _derive_direction_and_concentration(
            east, north, weight, min_concentration=0.0
        )

        assert np.isnan(direction.values[0])
        assert np.isnan(concentration.values[0])
        assert direction.values[1] == 0.0
        assert concentration.values[1] == 1.0
        assert direction.values[2] == 90.0
