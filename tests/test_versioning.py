"""Tests for the versioned reconciliation module."""

import cftime
import numpy as np
import pytest
import xarray as xr

from xcdat.versioning import (
    VersionConflict,
    VersionSegment,
    monthly_mean,
    monthly_provenance,
    reconcile_datasets,
    reconcile_versions,
    valid_duration,
    version_conflicts,
    version_provenance,
)

DAY = 86400.0


def _gregorian(year, month, day=1):
    return cftime.DatetimeGregorian(
        year, month, day, 0, 0, 0, 0, has_year_zero=False
    )


def _make_dataset(intervals, values, source, calendar="standard"):
    """Builds a dataset from (start, end) interval pairs and per-interval values.

    The time coordinate of each interval is its midpoint and ``time_bnds``
    stores the actual interval. Both are cftime objects of the same calendar.
    """
    times = np.array(
        [start + (end - start) / 2 for start, end in intervals], dtype=object
    )
    bounds = np.array(intervals, dtype=object)
    array = np.repeat(
        np.array(values, dtype=float).reshape(len(intervals), 1, 1), 2, axis=1
    ).repeat(2, axis=2)

    lat = xr.DataArray(
        [0.0, 1.0],
        dims="lat",
        attrs={"axis": "Y", "standard_name": "latitude", "units": "degrees_north"},
    )
    lon = xr.DataArray(
        [10.0, 20.0],
        dims="lon",
        attrs={"axis": "X", "standard_name": "longitude", "units": "degrees_east"},
    )
    ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), array)},
        coords={
            "time": ("time", times),
            "time_bnds": (("time", "bnds"), bounds),
            "lat": lat,
            "lon": lon,
        },
    )
    ds["time"].attrs.update(
        {
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": "time_bnds",
        }
    )
    ds["time_bnds"].attrs["xcdat_bounds"] = "True"
    ds.encoding["source"] = source
    ds.attrs["calendar"] = calendar
    return ds


def _monthly(intervals, values, source):
    return _make_dataset(intervals, values, source)


@pytest.fixture
def old_data():
    """An old version covering Jan-Apr 2000 with monthly values 1-4."""
    intervals = [
        (_gregorian(2000, 1), _gregorian(2000, 2)),
        (_gregorian(2000, 2), _gregorian(2000, 3)),
        (_gregorian(2000, 3), _gregorian(2000, 4)),
        (_gregorian(2000, 4), _gregorian(2000, 5)),
    ]
    return _monthly(intervals, [1.0, 2.0, 3.0, 4.0], "old.nc")


@pytest.fixture
def new_data():
    """A newer version correcting Feb and the first half of March."""
    intervals = [
        (_gregorian(2000, 2), _gregorian(2000, 3)),
        (_gregorian(2000, 3), _gregorian(2000, 3, 16)),
    ]
    return _monthly(intervals, [20.0, 30.0], "new.nc")


class TestReconcileDatasets:
    def test_overlap_resolved_by_explicit_version_and_uncovered_old_segment_kept(
        self, old_data
    ):
        new_intervals = [
            (_gregorian(2000, 2), _gregorian(2000, 3)),
            (_gregorian(2000, 3), _gregorian(2000, 3, 16)),
        ]
        new_data = _monthly(new_intervals, [20.0, 30.0], "new.nc")

        result = reconcile_datasets([old_data, new_data], [1, 2])

        segments = version_provenance(result)
        assert [segment.file for segment in segments] == [
            "old.nc",
            "new.nc",
            "new.nc",
            "old.nc",
            "old.nc",
        ]
        # The January and April values come from the old version; February is
        # corrected by the new version and the partial March segment is split.
        values = result["ts"].values[:, 0, 0]
        assert np.allclose(values, [1.0, 20.0, 30.0, 3.0, 4.0])

    def test_version_wins_regardless_of_input_order(self, old_data, new_data):
        forward = reconcile_datasets([old_data, new_data], [1, 2])
        backward = reconcile_datasets([new_data, old_data], [2, 1])

        assert forward["ts"].values.tolist() == backward["ts"].values.tolist()
        assert version_provenance(forward) == version_provenance(backward)
        assert version_conflicts(forward) == version_conflicts(backward)

    def test_duplicate_file_is_idempotent(self, old_data, new_data):
        single = reconcile_datasets([old_data, new_data], [1, 2])
        duplicated = reconcile_datasets([old_data, new_data, old_data], [1, 2, 1])

        assert single["ts"].values.tolist() == duplicated["ts"].values.tolist()
        assert version_provenance(single) == version_provenance(duplicated)
        assert version_conflicts(single) == version_conflicts(duplicated)

    def test_same_version_conflict_is_reported_and_excluded(self, old_data):
        conflicting = _make_dataset(
            [(_gregorian(2000, 2), _gregorian(2000, 3))], [99.0], "other.nc"
        )

        result = reconcile_datasets([old_data, conflicting], [1, 1])

        conflicts = version_conflicts(result)
        assert len(conflicts) == 1
        assert isinstance(conflicts[0], VersionConflict)
        assert conflicts[0].files == ("old.nc", "other.nc")
        assert conflicts[0].start == "2000-02-01T00:00:00"
        assert conflicts[0].end == "2000-03-01T00:00:00"
        assert conflicts[0].duration_seconds == pytest.approx(29 * DAY)

        # The conflicting February interval is excluded from the record.
        values = result["ts"].values[:, 0, 0]
        assert np.allclose(values, [1.0, 3.0, 4.0])
        assert [s.file for s in version_provenance(result)] == [
            "old.nc",
            "old.nc",
            "old.nc",
        ]

    def test_same_version_identical_values_are_not_a_conflict(self):
        left = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [5.0], "a.nc"
        )
        right = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [5.0], "b.nc"
        )

        result = reconcile_datasets([left, right], [1, 1])

        assert version_conflicts(result) == []
        assert np.allclose(result["ts"].values[:, 0, 0], [5.0])
        # The merged interval is attributed to one deterministic source.
        assert version_provenance(result)[0].file == "a.nc"

    def test_conflict_files_listed_in_sorted_order(self):
        later = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [2.0], "zeta.nc"
        )
        earlier = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [1.0], "alpha.nc"
        )

        result = reconcile_datasets([later, earlier], [1, 1])

        assert version_conflicts(result)[0].files == ("alpha.nc", "zeta.nc")

    def test_higher_version_does_not_conflict_with_lower_version(self):
        old = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [1.0], "old.nc"
        )
        new = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [9.0], "new.nc"
        )

        result = reconcile_datasets([old, new], [1, 2])

        assert version_conflicts(result) == []
        assert np.allclose(result["ts"].values[:, 0, 0], [9.0])
        assert version_provenance(result)[0].version == 2

    def test_string_versions_are_ordered_numerically(self):
        older = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [1.0], "older.nc"
        )
        newer = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [2.0], "newer.nc"
        )

        result = reconcile_datasets([older, newer], ["v1.2.0", "v1.10.0"])

        assert np.allclose(result["ts"].values[:, 0, 0], [2.0])

    def test_raises_if_versions_length_mismatch(self, old_data):
        with pytest.raises(ValueError, match="datasets but"):
            reconcile_datasets([old_data], [1, 2])

    def test_raises_if_grids_differ(self, old_data):
        other = old_data.copy()
        other = other.assign_coords(lat=[5.0, 6.0])
        other.encoding["source"] = "other.nc"

        with pytest.raises(ValueError, match="non-time coordinates differ"):
            reconcile_datasets([old_data, other], [1, 2])

    def test_raises_if_calendars_differ(self, old_data):
        noleap = _make_dataset(
            [
                (
                    cftime.DatetimeNoLeap(2000, 1, 1),
                    cftime.DatetimeNoLeap(2000, 2, 1),
                )
            ],
            [1.0],
            "noleap.nc",
        )

        with pytest.raises(ValueError, match="mixed time types or calendars"):
            reconcile_datasets([old_data, noleap], [1, 2])

    def test_static_and_bounds_variables_are_preserved(self, old_data, new_data):
        result = reconcile_datasets([old_data, new_data], [1, 2])

        assert "lat" in result.coords
        assert "lon" in result.coords
        assert result["time"].attrs["bounds"] == "time_bnds"
        assert "time_bnds" in result.variables

    def test_provenance_durations_sum_to_valid_duration(self, old_data, new_data):
        result = reconcile_datasets([old_data, new_data], [1, 2])

        total = sum(segment.duration_seconds for segment in version_provenance(result))
        jan_to_may = (
            _gregorian(2000, 5) - _gregorian(2000, 1)
        ).total_seconds()
        assert total == pytest.approx(jan_to_may)

    def test_all_conflicting_intervals_produce_empty_record(self):
        left = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [1.0], "a.nc"
        )
        right = _make_dataset(
            [(_gregorian(2000, 1), _gregorian(2000, 2))], [2.0], "b.nc"
        )

        result = reconcile_datasets([left, right], [1, 1])

        assert result.sizes["time"] == 0
        assert "ts" not in result
        assert version_provenance(result) == []
        assert len(version_conflicts(result)) == 1

        # The conflict is still reported by the monthly APIs.
        means = monthly_mean(result, "ts")
        assert means["valid_duration"].values.tolist() == [0.0]
        assert means["conflict_duration"].values.tolist() == [31 * DAY]


class TestMonthlyMean:
    def test_partial_month_is_weighted_by_actual_duration(self, old_data, new_data):
        result = reconcile_datasets([old_data, new_data], [1, 2])

        means = monthly_mean(result, "ts")

        march = [
            i for i, time in enumerate(means["time"].values) if time.month == 3
        ][0]
        # March is 15 days of version 2 (value 30) and 16 days of version 1
        # (value 3) in a 31 day month.
        expected = (15 * 30.0 + 16 * 3.0) / 31
        assert means["ts"].values[march, 0, 0] == pytest.approx(expected)
        assert means["valid_duration"].values[march] == pytest.approx(31 * DAY)

    def test_monthly_values_from_effective_intervals(self, old_data, new_data):
        result = reconcile_datasets([old_data, new_data], [1, 2])

        means = monthly_mean(result, "ts")

        assert np.allclose(means["ts"].values[:, 0, 0], [1.0, 20.0, 16.064516129, 4.0])
        assert np.allclose(
            means["valid_duration"].values,
            [31 * DAY, 29 * DAY, 31 * DAY, 30 * DAY],
        )
        assert np.allclose(means["conflict_duration"].values, 0.0)

    def test_monthly_mean_is_order_independent(self, old_data, new_data):
        forward = monthly_mean(
            reconcile_datasets([old_data, new_data], [1, 2]), "ts"
        )
        backward = monthly_mean(
            reconcile_datasets([new_data, old_data], [2, 1]), "ts"
        )

        xr.testing.assert_identical(forward, backward)

    def test_conflicting_month_has_no_mean_but_records_duration(self, old_data):
        conflicting = _make_dataset(
            [(_gregorian(2000, 2), _gregorian(2000, 3))], [99.0], "other.nc"
        )
        result = reconcile_datasets([old_data, conflicting], [1, 1])

        means = monthly_mean(result, "ts")

        feb = [i for i, time in enumerate(means["time"].values) if time.month == 2][0]
        assert np.isnan(means["ts"].values[feb, 0, 0])
        assert means["valid_duration"].values[feb] == pytest.approx(0.0)
        assert means["conflict_duration"].values[feb] == pytest.approx(29 * DAY)

    def test_valid_duration_matches_month_length(self):
        intervals = [(_gregorian(2000, 1), _gregorian(2000, 2))]
        ds = _monthly(intervals, [1.0], "single.nc")

        duration = valid_duration(ds)

        assert duration.name == "valid_duration"
        assert duration.values.tolist() == [31 * DAY]

    def test_infers_data_var_from_reconciliation_info(self, old_data, new_data):
        result = reconcile_datasets([old_data, new_data], [1, 2])

        inferred = monthly_mean(result)
        explicit = monthly_mean(result, "ts")

        assert inferred["ts"].equals(explicit["ts"])


class TestProvenance:
    def test_monthly_provenance_traces_each_month(self, old_data, new_data):
        result = reconcile_datasets([old_data, new_data], [1, 2])

        provenance = {entry.month: entry for entry in monthly_provenance(result)}

        assert provenance["2000-02"].segments[0].file == "new.nc"
        assert provenance["2000-02"].segments[0].version == 2
        assert provenance["2000-02"].valid_duration_seconds == pytest.approx(29 * DAY)
        # March is split between the two sources.
        march_files = {segment.file for segment in provenance["2000-03"].segments}
        assert march_files == {"new.nc", "old.nc"}
        assert provenance["2000-03"].valid_duration_seconds == pytest.approx(31 * DAY)

    def test_monthly_provenance_includes_conflict_durations(self, old_data):
        conflicting = _make_dataset(
            [(_gregorian(2000, 2), _gregorian(2000, 3))], [99.0], "other.nc"
        )
        result = reconcile_datasets([old_data, conflicting], [1, 1])

        provenance = {entry.month: entry for entry in monthly_provenance(result)}

        assert provenance["2000-02"].conflict_duration_seconds == pytest.approx(
            29 * DAY
        )
        assert provenance["2000-02"].valid_duration_seconds == pytest.approx(0.0)

    def test_segments_are_version_segments(self, old_data):
        result = reconcile_datasets([old_data], [3])

        segments = version_provenance(result)

        assert all(isinstance(segment, VersionSegment) for segment in segments)
        assert {segment.version for segment in segments} == {3}

    def test_no_info_returns_empty_lists(self):
        ds = xr.Dataset({"v": ("time", [1.0])}, coords={"time": [0]})

        assert version_provenance(ds) == []
        assert version_conflicts(ds) == []
        assert monthly_provenance(ds) == []


class TestReconcileVersionsIO:
    def _write(self, ds, path, version, units="days since 2000-01-01 00:00:00"):
        ds = ds.copy()
        ds.attrs["version"] = version
        ds["time"].encoding.update(
            {"units": units, "calendar": "standard", "dtype": "float64"}
        )
        ds["time_bnds"].encoding.update(
            {"units": units, "calendar": "standard", "dtype": "float64"}
        )
        ds.to_netcdf(path)

    def test_reads_version_from_global_attribute(self, tmp_path, old_data, new_data):
        self._write(old_data, tmp_path / "a.nc", 1)
        self._write(new_data, tmp_path / "b.nc", 2)

        result = reconcile_versions(tmp_path, data_var="ts", add_bounds=None)

        assert np.allclose(result["ts"].values[:, 0, 0], [1.0, 20.0, 30.0, 3.0, 4.0])
        assert {segment.version for segment in version_provenance(result)} == {1, 2}

    def test_version_mapping_resolves_by_basename(self, tmp_path, old_data, new_data):
        a = tmp_path / "a.nc"
        b = tmp_path / "b.nc"
        self._write(old_data, a, 1)
        self._write(new_data, b, 2)

        result = reconcile_versions(
            [a, b],
            data_var="ts",
            add_bounds=None,
            version={"a.nc": "v1", "b.nc": "v2"},
        )

        assert {segment.version for segment in version_provenance(result)} == {
            "v1",
            "v2",
        }

    def test_duplicate_paths_are_deduplicated(self, tmp_path, old_data, new_data):
        a = tmp_path / "a.nc"
        b = tmp_path / "b.nc"
        self._write(old_data, a, 1)
        self._write(new_data, b, 2)

        single = reconcile_versions([a, b], data_var="ts", add_bounds=None)
        duplicated = reconcile_versions(
            [a, b, a], data_var="ts", add_bounds=None
        )

        assert single["ts"].values.tolist() == duplicated["ts"].values.tolist()

    def test_summary_is_invariant_to_order_and_duplicates(
        self, tmp_path, old_data, new_data
    ):
        a = tmp_path / "a.nc"
        b = tmp_path / "b.nc"
        self._write(old_data, a, 1)
        self._write(new_data, b, 2)

        def summary(paths):
            ds = reconcile_versions(paths, data_var="ts", add_bounds=None)
            means = monthly_mean(ds, "ts")
            conflicts = version_conflicts(ds)
            return (
                means["ts"].values.tolist(),
                means["valid_duration"].values.tolist(),
                [(c.start, c.end, c.files) for c in conflicts],
            )

        baseline = summary([a, b])
        assert summary([b, a]) == baseline
        assert summary([a, b, a]) == baseline
        assert summary([b, a, b, a]) == baseline

    def test_raises_if_version_missing(self, tmp_path, old_data):
        a = tmp_path / "a.nc"
        encoded = old_data.copy()
        encoded["time"].encoding.update(
            {"units": "days since 2000-01-01 00:00:00", "calendar": "standard"}
        )
        encoded["time_bnds"].encoding.update(
            {"units": "days since 2000-01-01 00:00:00", "calendar": "standard"}
        )
        encoded.to_netcdf(a)

        with pytest.raises(ValueError, match="Could not determine an explicit version"):
            reconcile_versions(a, data_var="ts", add_bounds=None)
