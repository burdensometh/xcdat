"""Tests for variable-aware coordinate/bounds/mask resolution.

These tests cover merged datasets that contain two same-length grids (e.g.,
ocean and land variables). Each variable must use its own coordinates,
bounds, and mask; when two candidate axes are both plausible the conflict
must be reported, and an explicit selection must only affect the current
call.
"""

import numpy as np
import pytest
import xarray as xr

import xcdat
from xcdat.axis import AmbiguousAxisError, get_axis_coord
from xcdat.regridder.accessor import _get_input_grid


def _build_dual_grid_ds(with_coordinates_attr: bool = True) -> xr.Dataset:
    """Build a merged ocean/land dataset sharing the (lat, lon) dims.

    The land grid uses index coordinates ``lat``/``lon``; the ocean grid is
    stored as same-length auxiliary coordinates ``olat``/``olon``. Both have
    their own bounds.
    """
    lat = xr.DataArray(
        np.arange(-60, 61, 30),
        dims="lat",
        name="lat",
        attrs={
            "axis": "Y",
            "standard_name": "latitude",
            "units": "degrees_north",
            "bounds": "lat_bnds",
        },
    )
    lon = xr.DataArray(
        np.arange(-120, 61, 60),
        dims="lon",
        name="lon",
        attrs={
            "axis": "X",
            "standard_name": "longitude",
            "units": "degrees_east",
            "bounds": "lon_bnds",
        },
    )
    olat = xr.DataArray(
        np.arange(-80, 81, 40),
        dims="lat",
        name="olat",
        attrs={
            "axis": "Y",
            "standard_name": "latitude",
            "units": "degrees_north",
            "bounds": "olat_bnds",
        },
    )
    olon = xr.DataArray(
        np.arange(-150, 31, 60),
        dims="lon",
        name="olon",
        attrs={
            "axis": "X",
            "standard_name": "longitude",
            "units": "degrees_east",
            "bounds": "olon_bnds",
        },
    )

    rng = np.random.default_rng(3)
    tos = xr.DataArray(
        rng.uniform(273, 303, (5, 4)), dims=("lat", "lon"), name="tos"
    )
    tas = xr.DataArray(
        rng.uniform(260, 310, (5, 4)), dims=("lat", "lon"), name="tas"
    )

    if with_coordinates_attr:
        tos.attrs["coordinates"] = "olat olon"
        tas.attrs["coordinates"] = "lat lon"

    return xr.Dataset(
        {
            "tos": tos,
            "tas": tas,
            "lat_bnds": xr.DataArray(
                np.column_stack([lat - 15, lat + 15]), dims=("lat", "bnds")
            ),
            "lon_bnds": xr.DataArray(
                np.column_stack([lon - 30, lon + 30]), dims=("lon", "bnds")
            ),
            "olat_bnds": xr.DataArray(
                np.column_stack([olat - 20, olat + 20]), dims=("lat", "bnds")
            ),
            "olon_bnds": xr.DataArray(
                np.column_stack([olon - 30, olon + 30]), dims=("lon", "bnds")
            ),
        },
        coords={"lat": lat, "lon": lon, "olat": olat, "olon": olon},
    )


def _split_single_grid(
    ds: xr.Dataset, var: str, coord_names: list[str], bnd_names: list[str]
) -> xr.Dataset:
    """Reconstruct a realistic standalone single-grid file."""
    single = ds[[var] + coord_names + bnd_names].copy()
    single = single.drop_vars(
        [c for c in single.coords if c not in coord_names]
    )

    swap = {
        dim: coord
        for coord in coord_names
        for dim in single[coord].dims
        if dim != coord and dim in single.dims
    }
    if swap:
        single = single.swap_dims(swap)

    return single


class TestAxisResolver:
    def test_reports_conflict_when_two_coords_plausible(self):
        ds = _build_dual_grid_ds(with_coordinates_attr=False)

        with pytest.raises(AmbiguousAxisError, match="more than one coordinate"):
            get_axis_coord(ds["tos"], "X")
        with pytest.raises(AmbiguousAxisError):
            get_axis_coord(ds["tas"], "Y")

    def test_coordinates_attribute_selects_variable_grid(self):
        ds = _build_dual_grid_ds()

        assert get_axis_coord(ds["tos"], "X").name == "olon"
        assert get_axis_coord(ds["tos"], "Y").name == "olat"
        assert get_axis_coord(ds["tas"], "X").name == "lon"
        assert get_axis_coord(ds["tas"], "Y").name == "lat"

    def test_explicit_selection_is_call_scoped_only(self):
        ds = _build_dual_grid_ds(with_coordinates_attr=False)

        overridden = get_axis_coord(ds["tos"], "X", coord_key="lon")
        assert overridden.name == "lon"

        # A subsequent call without the override is still ambiguous.
        with pytest.raises(AmbiguousAxisError):
            get_axis_coord(ds["tos"], "X")

    def test_explicit_selection_validation(self):
        ds = _build_dual_grid_ds()

        with pytest.raises(KeyError):
            get_axis_coord(ds["tos"], "X", coord_key="missing")
        with pytest.raises(ValueError, match="not mapped to the 'Y' axis"):
            get_axis_coord(ds["tos"], "Y", coord_key="olon")


class TestSpatialAverageMultiGrid:
    @pytest.mark.parametrize(
        "var,coords,bnds",
        [
            ("tos", ["olat", "olon"], ["olat_bnds", "olon_bnds"]),
            ("tas", ["lat", "lon"], ["lat_bnds", "lon_bnds"]),
        ],
    )
    def test_merged_matches_split_single_grid(self, var, coords, bnds):
        ds = _build_dual_grid_ds()

        merged = ds.spatial.average(var)[var]
        reference = _split_single_grid(ds, var, coords, bnds).spatial.average(
            var
        )[var]

        np.testing.assert_allclose(float(merged), float(reference), rtol=1e-12)

    def test_regional_average_uses_own_bounds(self):
        ds = _build_dual_grid_ds()

        merged = ds.spatial.average(
            "tos", lat_bounds=(-30, 30), lon_bounds=(-120, 0)
        )["tos"]
        reference = _split_single_grid(
            ds, "tos", ["olat", "olon"], ["olat_bnds", "olon_bnds"]
        ).spatial.average(
            "tos", lat_bounds=(-30, 30), lon_bounds=(-120, 0)
        )["tos"]

        np.testing.assert_allclose(float(merged), float(reference), rtol=1e-12)

    def test_ambiguity_raised_then_resolvable_per_call(self):
        ds = _build_dual_grid_ds(with_coordinates_attr=False)

        with pytest.raises(AmbiguousAxisError):
            ds.spatial.average("tos")

        out = ds.spatial.average(
            "tos", coord_keys={"X": "olon", "Y": "olat"}
        )["tos"]
        reference = _split_single_grid(
            ds, "tos", ["olat", "olon"], ["olat_bnds", "olon_bnds"]
        ).spatial.average("tos")["tos"]

        np.testing.assert_allclose(float(out), float(reference), rtol=1e-12)

    def test_explicit_override_does_not_change_other_variable(self):
        ds = _build_dual_grid_ds(with_coordinates_attr=False)

        # Overriding tos to the ocean grid must not influence tas, which still
        # raises because its own conflict is unresolved.
        ds.spatial.average("tos", coord_keys={"X": "olon", "Y": "olat"})
        with pytest.raises(AmbiguousAxisError):
            ds.spatial.average("tas")


class TestRegridderMultiGrid:
    @pytest.mark.parametrize(
        "var,expected",
        [("tos", ["olat", "olon"]), ("tas", ["lat", "lon"])],
    )
    def test_input_grid_uses_variable_coords_and_bounds(self, var, expected):
        ds = _build_dual_grid_ds()

        grid = _get_input_grid(ds, var, ["X", "Y"])

        assert sorted(grid.coords) == expected
        assert {f"{expected[0]}_bnds", f"{expected[1]}_bnds"} <= set(
            grid.data_vars
        )

    @pytest.mark.parametrize("tool", ["regrid2", "xesmf"])
    def test_horizontal_matches_split_single_grid(self, tool):
        ds = _build_dual_grid_ds()
        output_grid = xcdat.create_uniform_grid(-89, 89, 4.0, -179, 179, 5.0)
        kwargs = {} if tool == "regrid2" else {"method": "bilinear"}

        for var, coords, bnds in (
            ("tos", ["olat", "olon"], ["olat_bnds", "olon_bnds"]),
            ("tas", ["lat", "lon"], ["lat_bnds", "lon_bnds"]),
        ):
            merged = ds.regridder.horizontal(
                var, output_grid, tool=tool, **kwargs
            )[var]
            split = _split_single_grid(ds, var, coords, bnds).regridder.horizontal(
                var, output_grid, tool=tool, **kwargs
            )[var]

            np.testing.assert_allclose(
                merged.values, split.values, rtol=1e-5, atol=1e-6
            )

    def test_ambiguity_raised_then_override_works(self):
        ds = _build_dual_grid_ds(with_coordinates_attr=False)
        output_grid = xcdat.create_uniform_grid(-89, 89, 4.0, -179, 179, 5.0)

        with pytest.raises(AmbiguousAxisError):
            _get_input_grid(ds, "tos", ["X", "Y"])

        grid = _get_input_grid(
            ds,
            "tos",
            ["X", "Y"],
            coord_keys={"X": "olon", "Y": "olat"},
        )
        assert sorted(grid.coords) == ["olat", "olon"]

    def test_existing_mask_is_retained_on_resolved_grid(self):
        ds = _build_dual_grid_ds()
        ds["mask"] = xr.DataArray(np.ones((5, 4)), dims=("lat", "lon"))

        grid = _get_input_grid(ds, "tas", ["X", "Y"])
        assert "mask" in grid.data_vars


def _build_separate_dim_ds() -> xr.Dataset:
    """Two grids with distinct dimension and coordinate names."""
    olat = xr.DataArray(
        np.arange(-80, 81, 40),
        dims="olat",
        attrs={
            "axis": "Y",
            "standard_name": "latitude",
            "units": "degrees_north",
            "bounds": "olat_bnds",
        },
    )
    olon = xr.DataArray(
        np.arange(-150, 31, 60),
        dims="olon",
        attrs={
            "axis": "X",
            "standard_name": "longitude",
            "units": "degrees_east",
            "bounds": "olon_bnds",
        },
    )
    lat = xr.DataArray(
        np.arange(-60, 61, 30),
        dims="lat",
        attrs={
            "axis": "Y",
            "standard_name": "latitude",
            "units": "degrees_north",
            "bounds": "lat_bnds",
        },
    )
    lon = xr.DataArray(
        np.arange(-120, 61, 60),
        dims="lon",
        attrs={
            "axis": "X",
            "standard_name": "longitude",
            "units": "degrees_east",
            "bounds": "lon_bnds",
        },
    )
    rng = np.random.default_rng(5)
    return xr.Dataset(
        {
            "tos": xr.DataArray(
                rng.uniform(273, 303, (5, 4)), dims=("olat", "olon"), name="tos"
            ),
            "tas": xr.DataArray(
                rng.uniform(260, 310, (5, 4)), dims=("lat", "lon"), name="tas"
            ),
            "olat_bnds": xr.DataArray(
                np.column_stack([olat - 20, olat + 20]), dims=("olat", "bnds")
            ),
            "olon_bnds": xr.DataArray(
                np.column_stack([olon - 30, olon + 30]), dims=("olon", "bnds")
            ),
            "lat_bnds": xr.DataArray(
                np.column_stack([lat - 15, lat + 15]), dims=("lat", "bnds")
            ),
            "lon_bnds": xr.DataArray(
                np.column_stack([lon - 30, lon + 30]), dims=("lon", "bnds")
            ),
            # An ocean-only mask; it must not be attached to the land grid.
            "mask": xr.DataArray(np.ones((5, 4)), dims=("olat", "olon")),
        },
        coords={"olat": olat, "olon": olon, "lat": lat, "lon": lon},
    )


class TestSeparateDimensionGrids:
    def test_input_grids_are_distinct(self):
        ds = _build_separate_dim_ds()

        ocean = _get_input_grid(ds, "tos", ["X", "Y"])
        land = _get_input_grid(ds, "tas", ["X", "Y"])

        assert sorted(ocean.coords) == ["olat", "olon"]
        assert sorted(land.coords) == ["lat", "lon"]
        assert "mask" in ocean.data_vars
        assert "mask" not in land.data_vars

    @pytest.mark.parametrize("var", ["tos", "tas"])
    def test_average_matches_split_single_grid(self, var):
        ds = _build_separate_dim_ds()
        coords = ["olat", "olon"] if var == "tos" else ["lat", "lon"]
        bnds = [f"{coords[0]}_bnds", f"{coords[1]}_bnds"]

        merged = ds.spatial.average(var)[var]
        reference = _split_single_grid(ds, var, coords, bnds).spatial.average(
            var
        )[var]

        np.testing.assert_allclose(float(merged), float(reference), rtol=1e-12)

    @pytest.mark.parametrize("tool", ["regrid2", "xesmf"])
    def test_horizontal_matches_split_single_grid(self, tool):
        ds = _build_separate_dim_ds()
        output_grid = xcdat.create_uniform_grid(-89, 89, 4.0, -179, 179, 5.0)
        kwargs = {} if tool == "regrid2" else {"method": "bilinear"}

        for var, coords, bnds in (
            ("tos", ["olat", "olon"], ["olat_bnds", "olon_bnds"]),
            ("tas", ["lat", "lon"], ["lat_bnds", "lon_bnds"]),
        ):
            merged = ds.regridder.horizontal(
                var, output_grid, tool=tool, **kwargs
            )[var]
            split = _split_single_grid(ds, var, coords, bnds).regridder.horizontal(
                var, output_grid, tool=tool, **kwargs
            )[var]

            np.testing.assert_allclose(
                merged.values, split.values, rtol=1e-5, atol=1e-6
            )
