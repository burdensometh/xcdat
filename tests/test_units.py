"""Tests for the unit harmonization module (`xcdat.units`)."""

import numpy as np
import pytest
import xarray as xr
from dask.array.core import Array as DaskArray

from xcdat.units import (
    SOURCE_UNITS_ATTR,
    TARGET_UNITS_ATTR,
    SUPPORTED_TEMPERATURE_UNITS,
    build_source_units_record,
    convert_temperature_dataarray,
    harmonize_dataset_units,
    merge_source_units_records,
    normalize_temperature_unit,
    validate_harmonize_units,
)


def _temperature_da(
    values: list[float], units: str, fill: float | None = None
) -> xr.DataArray:
    attrs: dict[str, object] = {"units": units, "long_name": "temperature"}
    if fill is not None:
        attrs["_FillValue"] = np.float64(fill)
        attrs["valid_range"] = np.array([0.0, 400.0])
    return xr.DataArray(np.array(values, dtype="float64"), dims="time", attrs=attrs)


class TestNormalizeTemperatureUnit:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("K", "K"),
            ("k", "K"),
            ("kelvin", "K"),
            ("Kelvin", "K"),
            ("degrees_Kelvin", "K"),
            ("degC", "degC"),
            ("celsius", "degC"),
            ("C", "degC"),
            ("degree_celsius", "degC"),
            ("degF", "degF"),
            ("Fahrenheit", "degF"),
            ("f", "degF"),
        ],
    )
    def test_normalizes_aliases(self, label, expected):
        assert normalize_temperature_unit(label) == expected

    def test_strips_whitespace(self):
        assert normalize_temperature_unit("  kelvin ") == "K"

    @pytest.mark.parametrize("label", ["rankine", "banana", "", "Pa"])
    def test_raises_for_unsupported_unit(self, label):
        with pytest.raises(ValueError, match="Unsupported temperature unit"):
            normalize_temperature_unit(label)

    def test_raises_for_non_string(self):
        with pytest.raises(ValueError, match="Expected a string unit label"):
            normalize_temperature_unit(1)  # type: ignore

    def test_supported_units_constant(self):
        assert set(SUPPORTED_TEMPERATURE_UNITS) == {"K", "degC", "degF"}


class TestConvertTemperatureDataArray:
    def test_celsius_to_kelvin(self):
        da = _temperature_da([0.0, 100.0], "degC")
        result = convert_temperature_dataarray(da, "K")

        np.testing.assert_allclose(result.values, [273.15, 373.15])
        assert result.attrs["units"] == "K"

    def test_kelvin_to_celsius(self):
        da = _temperature_da([273.15, 373.15], "K")
        result = convert_temperature_dataarray(da, "degC")

        np.testing.assert_allclose(result.values, [0.0, 100.0])
        assert result.attrs["units"] == "degC"

    def test_celsius_to_fahrenheit(self):
        da = _temperature_da([0.0, 100.0], "degC")
        result = convert_temperature_dataarray(da, "degF")

        np.testing.assert_allclose(result.values, [32.0, 212.0])

    def test_fahrenheit_to_kelvin(self):
        da = _temperature_da([32.0, 212.0], "degF")
        result = convert_temperature_dataarray(da, "K")

        np.testing.assert_allclose(result.values, [273.15, 373.15])

    def test_kelvin_to_fahrenheit(self):
        da = _temperature_da([273.15], "K")
        result = convert_temperature_dataarray(da, "degF")

        np.testing.assert_allclose(result.values, [32.0])

    def test_fahrenheit_to_celsius(self):
        da = _temperature_da([32.0, 212.0], "degF")
        result = convert_temperature_dataarray(da, "degC")

        np.testing.assert_allclose(result.values, [0.0, 100.0])

    def test_same_unit_is_value_preserving(self):
        da = _temperature_da([1.0, 2.0], "K")
        result = convert_temperature_dataarray(da, "K")

        np.testing.assert_array_equal(result.values, da.values)
        assert result.attrs["units"] == "K"

    def test_missing_values_are_not_transformed(self):
        # -999.0 is the file's fill sentinel; after mask_and_scale it is NaN.
        da = _temperature_da([0.0, np.nan, 10.0], "degC")
        result = convert_temperature_dataarray(da, "K")

        assert np.isnan(result.values[1])
        np.testing.assert_allclose(
            result.values[[0, 2]], [273.15, 283.15]
        )

    def test_is_lazy_for_dask_backed_arrays(self):
        da = _temperature_da([0.0, 10.0], "degC").chunk()
        result = convert_temperature_dataarray(da, "K")

        assert isinstance(result.data, DaskArray)
        np.testing.assert_allclose(result.values, [273.15, 283.15])

    def test_writes_source_and_target_units(self):
        da = _temperature_da([0.0], "degC")
        result = convert_temperature_dataarray(da, "K")

        assert result.attrs[SOURCE_UNITS_ATTR] == "degC"
        assert result.attrs[TARGET_UNITS_ATTR] == "K"
        assert result.attrs["units"] == "K"

    def test_drops_encoding_specific_attrs(self):
        da = _temperature_da([0.0, np.nan], "degC", fill=-999.0)
        result = convert_temperature_dataarray(da, "K")

        assert "_FillValue" not in result.attrs
        assert "valid_range" not in result.attrs
        # Unrelated descriptive attributes are retained.
        assert result.attrs["long_name"] == "temperature"

    def test_does_not_mutate_input(self):
        da = _temperature_da([0.0], "degC", fill=-999.0)
        original_attrs = dict(da.attrs)
        convert_temperature_dataarray(da, "K")

        assert da.attrs == original_attrs

    def test_raises_when_units_missing(self):
        da = xr.DataArray([1.0], dims="time")
        with pytest.raises(ValueError, match="has no 'units' attribute"):
            convert_temperature_dataarray(da, "K")

    def test_raises_for_unknown_source_unit(self):
        da = _temperature_da([1.0], "banana")
        with pytest.raises(ValueError, match="Unsupported temperature unit"):
            convert_temperature_dataarray(da, "K")


class TestHarmonizeDatasetUnits:
    def test_converts_listed_variable(self):
        ds = xr.Dataset({"tas": _temperature_da([0.0, 10.0], "degC")})
        result = harmonize_dataset_units(ds, {"tas": "K"})

        np.testing.assert_allclose(result["tas"].values, [273.15, 283.15])
        assert result["tas"].attrs["units"] == "K"

    def test_skips_variables_absent_from_dataset(self):
        ds = xr.Dataset({"tas": _temperature_da([0.0], "K")})
        # "other" exists only in a different source file; no error here.
        result = harmonize_dataset_units(ds, {"tas": "K", "other": "K"})

        assert "other" not in result.data_vars

    def test_unknown_unit_error_names_variable_and_source(self):
        ds = xr.Dataset({"tas": _temperature_da([1.0], "banana")})
        ds.encoding["source"] = "/data/provider_b/file.nc"

        with pytest.raises(ValueError) as exc:
            harmonize_dataset_units(ds, {"tas": "K"})

        message = str(exc.value)
        assert "tas" in message
        assert "/data/provider_b/file.nc" in message
        assert "banana" in message

    def test_missing_unit_error_names_variable_and_source(self):
        ds = xr.Dataset({"tas": xr.DataArray([1.0], dims="time")})
        ds.encoding["source"] = "/data/no_units.nc"

        with pytest.raises(ValueError) as exc:
            harmonize_dataset_units(ds, {"tas": "K"})

        assert "tas" in str(exc.value)
        assert "/data/no_units.nc" in str(exc.value)

    def test_missing_value_uses_source_fill_encoding(self):
        # Simulates post mask_and_scale: each provider's distinct sentinel has
        # already become NaN before harmonization runs.
        da_c = xr.DataArray(
            [0.0, np.nan, 10.0], dims="time", attrs={"units": "degC"}
        )
        ds = xr.Dataset({"tas": da_c})
        result = harmonize_dataset_units(ds, {"tas": "K"})

        assert np.isnan(result["tas"].values[1])
        np.testing.assert_allclose(result["tas"].values[[0, 2]], [273.15, 283.15])


class TestSourceUnitsRecords:
    def test_record_captures_file_and_unit(self):
        ds = xr.Dataset({"tas": _temperature_da([0.0], "degC")})
        ds.encoding["source"] = "/data/a.nc"

        record = build_source_units_record(ds, {"tas": "K"})

        assert record == {"tas": "/data/a.nc: degC"}

    def test_record_marks_unrecognized_unit_without_raising(self):
        ds = xr.Dataset({"tas": _temperature_da([0.0], "banana")})
        ds.encoding["source"] = "/data/b.nc"

        record = build_source_units_record(ds, {"tas": "K"})

        assert record == {"tas": "/data/b.nc: unrecognized (banana)"}

    def test_record_skips_absent_variable(self):
        ds = xr.Dataset({"tas": _temperature_da([0.0], "K")})
        ds.encoding["source"] = "/data/a.nc"

        record = build_source_units_record(ds, {"other": "K"})

        assert record == {}

    def test_merged_record_is_order_independent(self):
        records = [
            {"tas": "/data/a.nc: degC"},
            {"tas": "/data/b.nc: K"},
        ]
        merged_forward = merge_source_units_records(records)
        merged_reversed = merge_source_units_records(list(reversed(records)))

        expected = {"tas_source_units": "/data/a.nc: degC; /data/b.nc: K"}
        assert merged_forward == expected
        assert merged_reversed == expected


class TestValidateHarmonizeUnits:
    def test_accepts_valid_mapping(self):
        validate_harmonize_units({"tas": "K", "tos": "degC"})

    def test_raises_for_bad_target_unit(self):
        with pytest.raises(ValueError, match="Unsupported temperature unit"):
            validate_harmonize_units({"tas": "rankine"})

    def test_raises_for_non_string_key(self):
        with pytest.raises(ValueError, match="variable name strings"):
            validate_harmonize_units({1: "K"})  # type: ignore
