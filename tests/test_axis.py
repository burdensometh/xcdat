import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.axis import (
    CFAxisKey,
    center_times,
    get_coords_by_name,
    get_dim_coords,
    get_dim_keys,
    swap_lon_axis,
)


class TestGetDimKeys:
    def test_raises_error_if_dim_name_is_not_valid(self):
        ds = xr.Dataset(
            coords={
                "invalid_lat_shortname": xr.DataArray(
                    data=np.ones(3), dims="invalid_lat_shortname"
                )
            }
        )

        with pytest.raises(KeyError):
            get_dim_keys(ds, "Y")

    def test_returns_dim_name(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3), dims="lat", attrs={"standard_name": "latitude"}
                )
            }
        )

        dim = get_dim_keys(ds, "Y")

        assert dim == "lat"

    def test_returns_dim_names(self):
        ds = xr.Dataset(
            coords={
                "ilev": xr.DataArray(data=np.ones(3), dims="ilev", attrs={"axis": "Z"}),
                "lev": xr.DataArray(data=np.ones(3), dims="lev", attrs={"axis": "Z"}),
            }
        )

        dim = get_dim_keys(ds, "Z")

        assert dim == ["ilev", "lev"]


class TestGetDimCoords:
    @pytest.fixture(autouse=True)
    def setup(self):
        # A dataset with "axis" attr set on all dim coord vars.
        self.ds_axis = xr.Dataset(
            data_vars={
                "hyai": xr.DataArray(
                    name="hyai",
                    data=np.ones(3),
                    dims="ilev",
                    attrs={"long_name": "hybrid A coefficient at layer interfaces"},
                ),
                "hyam": xr.DataArray(
                    name="hyam",
                    data=np.ones(3),
                    dims="lev",
                    attrs={"long_name": "hybrid B coefficient at layer interfaces"},
                ),
            },
            coords={
                "ilev": xr.DataArray(
                    data=np.ones(3),
                    dims="ilev",
                    attrs={"axis": "Z"},
                ),
                "lev": xr.DataArray(
                    data=np.ones(3),
                    dims="lev",
                    attrs={"axis": "Z"},
                ),
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y"},
                ),
            },
        )

        # A dataset with "standard_name" attr set on all dim coord vars.
        self.ds_sn = xr.Dataset(
            data_vars={
                "hyai": xr.DataArray(
                    name="hyai",
                    data=np.ones(3),
                    dims="ilev",
                    attrs={"long_name": "hybrid A coefficient at layer interfaces"},
                ),
                "hyam": xr.DataArray(
                    name="hyam",
                    data=np.ones(3),
                    dims="lev",
                    attrs={"long_name": "hybrid B coefficient at layer interfaces"},
                ),
            },
            coords={
                "ilev": xr.DataArray(
                    data=np.ones(3),
                    dims="ilev",
                    attrs={
                        "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"
                    },
                ),
                "lev": xr.DataArray(
                    data=np.ones(3),
                    dims="lev",
                    attrs={
                        "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"
                    },
                ),
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"standard_name": "latitude"},
                ),
            },
        )

    def test_raises_error_if_dim_does_not_exist(self):
        ds = xr.Dataset()
        dims: CFAxisKey = ["X", "Y", "T", "Z"]  # type: ignore

        for dim in dims:
            with pytest.raises(KeyError):
                get_dim_coords(ds, dim)  # type: ignore

    def test_raises_error_if_axis_or_standard_name_is_not_set_or_dim_name_is_not_valid(
        self,
    ):
        ds = xr.Dataset(
            coords={
                "invalid_lat_shortname": xr.DataArray(
                    data=np.ones(3), dims="invalid_lat_shortname"
                )
            }
        )

        with pytest.raises(KeyError):
            get_dim_coords(ds, "Y")

    @pytest.mark.parametrize("multidim", [False, True])
    def test_raises_error_if_a_dataarray_has_multiple_dims_for_the_same_axis(
        self, multidim
    ):
        da = xr.DataArray(
            coords={
                "ilev": xr.DataArray(
                    data=np.ones(3),
                    dims="ilev",
                    attrs={"axis": "Z"},
                ),
                "lev": xr.DataArray(
                    data=np.ones(3),
                    dims="lev",
                    attrs={"axis": "Z"},
                ),
            },
            dims=["ilev", "lev"],
        )

        with pytest.raises(ValueError):
            get_dim_coords(da, "Z", multidim=multidim)

    def test_raises_error_if_multidimensional_coords_are_only_present_for_an_axis(self):
        lat = xr.DataArray(
            data=np.array([[0, 1, 2], [3, 4, 5]]),
            dims=["placeholder_1", "placeholder_2"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        ds = xr.Dataset(coords={"lat": lat})

        with pytest.raises(KeyError):
            get_dim_coords(ds, "Y")

    @pytest.mark.parametrize("multidim", [False, True])
    def test_returns_dataset_dimension_coordinate_vars_using_common_var_names(
        self,
        multidim,
    ):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(data=np.ones(3), dims="lat"),
                "lon": xr.DataArray(data=np.ones(3), dims="lon"),
                "time": xr.DataArray(data=np.ones(3), dims="time"),
                "lev": xr.DataArray(data=np.ones(3), dims="lev"),
            }
        )

        result = get_dim_coords(ds, "X", multidim=multidim)
        xr.testing.assert_identical(result, ds["lon"])

        result = get_dim_coords(ds, "Y", multidim=multidim)
        xr.testing.assert_identical(result, ds["lat"])

        result = get_dim_coords(ds, "T", multidim=multidim)
        xr.testing.assert_identical(result, ds["time"])

        result = get_dim_coords(ds, "Z", multidim=multidim)
        xr.testing.assert_identical(result, ds["lev"])

    @pytest.mark.parametrize("multidim", [False, True])
    def test_returns_dataset_dimension_coordinate_vars_using_axis_attr(self, multidim):
        # For example, E3SM datasets might have "ilev" and "lev" dimensions
        # with the dim coord var attr "axis" both mapped to "Z".
        result = get_dim_coords(self.ds_axis, "Z", multidim=multidim)
        expected = xr.Dataset(
            coords={"ilev": self.ds_axis.ilev, "lev": self.ds_axis.lev}
        )

        xr.testing.assert_identical(result, expected)

    def test_returns_dataset_dimension_coordinate_vars_using_standard_name_attr(self):
        # For example, E3SM datasets might have "ilev" and "lev" dimensions
        # with the dim coord var attr "standard_name" both mapped to
        # "atmosphere_hybrid_sigma_pressure_coordinate".
        result = get_dim_coords(self.ds_sn, "Z")
        expected = xr.Dataset(coords={"ilev": self.ds_sn.ilev, "lev": self.ds_sn.lev})

        xr.testing.assert_identical(result, expected)

    def test_returns_dataarray_dimension_coordinate_var_using_axis_attr(self):
        result = get_dim_coords(self.ds_axis.hyai, "Z")
        expected = self.ds_axis.ilev

        assert result.identical(expected)

        result = get_dim_coords(self.ds_axis.hyam, "Z")
        expected = self.ds_axis.lev

        assert result.identical(expected)

    def test_returns_dataarray_dimension_coordinate_var_using_standard_name_attr(self):
        result = get_dim_coords(self.ds_sn.hyai, "Z")
        expected = self.ds_sn.ilev

        assert result.identical(expected)

        result = get_dim_coords(self.ds_sn.hyam, "Z")
        expected = self.ds_sn.lev

        assert result.identical(expected)

    def test_multidim_finds_coords_only_in_cf_axes(self):
        """Test that multidim=True finds coords discoverable via cf axes."""
        # Create a dataset where 'lat' has axis="Y" attribute but no
        # standard_name or recognized coordinate name that cf_xarray would
        # put in obj.cf.coordinates.
        lat = xr.DataArray(
            data=np.array([[0, 1], [2, 3]]),
            dims=["y", "x"],
            attrs={"axis": "Y"},
        )
        lon = xr.DataArray(
            data=np.array([[10, 11], [12, 13]]),
            dims=["y", "x"],
            attrs={"axis": "X"},
        )
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        # Without multidim, this would raise KeyError because lat/lon are
        # multidimensional and not in indexes.
        with pytest.raises(KeyError):
            get_dim_coords(ds, "Y", multidim=False)

        # With multidim=True, it should find lat via cf axes.
        result = get_dim_coords(ds, "Y", multidim=True)
        xr.testing.assert_identical(result, ds["lat"])

    def test_multidim_ignores_scalar_axis_candidate(self):
        ds = xr.Dataset(
            coords={
                "plev": (
                    "plev",
                    [85000.0, 50000.0],
                    {
                        "axis": "Z",
                        "standard_name": "air_pressure",
                        "units": "Pa",
                        "positive": "down",
                    },
                ),
                "height": ((), 10.0, {"axis": "Z", "standard_name": "height"}),
            }
        )
        xr.testing.assert_identical(get_dim_coords(ds, "Z", multidim=True), ds.plev)

        with pytest.raises(KeyError):
            get_dim_coords(ds.drop_vars("plev"), "Z", multidim=True)

    def test_multidim_preserves_length_one_dimension_coordinate(self):
        ds = xr.Dataset(coords={"plev": ("plev", [85000.0], {"axis": "Z"})})
        xr.testing.assert_identical(get_dim_coords(ds, "Z", multidim=True), ds.plev)

    @pytest.mark.parametrize("axis, name", [("X", "longitude"), ("Y", "latitude")])
    def test_multidim_finds_common_long_names_without_cf_attrs(self, axis, name):
        ds = xr.Dataset(coords={name: (name, [0.0, 10.0])})
        xr.testing.assert_identical(get_dim_coords(ds, axis, multidim=True), ds[name])

    def test_multidim_finds_mixed_cf_metadata(self):
        ds = xr.Dataset(
            coords={
                "nav_lat": (("y", "x"), [[0, 1], [2, 3]], {"axis": "Y"}),
                "nav_lon": (
                    ("y", "x"),
                    [[10, 11], [12, 13]],
                    {"standard_name": "longitude"},
                ),
            }
        )
        xr.testing.assert_identical(get_dim_coords(ds, "Y", multidim=True), ds.nav_lat)
        xr.testing.assert_identical(get_dim_coords(ds, "X", multidim=True), ds.nav_lon)

    @pytest.mark.parametrize("cf_attrs", [False, True])
    @pytest.mark.parametrize(
        "lat_name, lon_name", [("lat", "lon"), ("latitude", "longitude")]
    )
    def test_multidim_prefers_physical_coords_over_axis_labelled_indexes(
        self, cf_attrs, lat_name, lon_name
    ):
        ds = xr.Dataset(
            coords={
                "rlat": (
                    "rlat",
                    [0.0, 1.0],
                    {"axis": "Y", "standard_name": "grid_latitude"},
                ),
                "rlon": (
                    "rlon",
                    [0.0, 1.0],
                    {"axis": "X", "standard_name": "grid_longitude"},
                ),
                "lat": (
                    ("rlat", "rlon"),
                    [[0.0, 0.0], [1.0, 1.0]],
                    {"standard_name": "latitude"},
                ),
                "lon": (
                    ("rlat", "rlon"),
                    [[0.0, 1.0], [0.0, 1.0]],
                    {"standard_name": "longitude"},
                ),
            }
        )
        if not cf_attrs:
            ds.lat.attrs = {}
            ds.lon.attrs = {}
        ds = ds.rename({"lat": lat_name, "lon": lon_name})
        for obj in (
            ds,
            xr.DataArray(np.ones((2, 2)), dims=("rlat", "rlon"), coords=ds.coords),
        ):
            xr.testing.assert_identical(
                get_dim_coords(obj, "Y", multidim=True), ds[lat_name]
            )
            xr.testing.assert_identical(
                get_dim_coords(obj, "X", multidim=True), ds[lon_name]
            )

    def test_multidim_prefers_cf_coords_over_common_names(self):
        ds = xr.Dataset(
            coords={
                "nav_lat": (
                    ("y", "x"),
                    [[0.0, 0.0], [1.0, 1.0]],
                    {"standard_name": "latitude"},
                ),
                "lat": ("y", [0.0, 1.0]),
                "x": ("x", [0.0, 1.0]),
            }
        )
        xr.testing.assert_identical(get_dim_coords(ds, "Y", multidim=True), ds.nav_lat)

    def test_multidim_name_preference_preserves_ambiguity(self):
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=("y", "x"),
            coords={
                "lat": (("y", "x"), [[0.0, 0.0], [1.0, 1.0]]),
                "latitude": (("y", "x"), [[2.0, 2.0], [3.0, 3.0]]),
            },
        )
        with pytest.raises(ValueError, match="more than one dimension"):
            get_dim_coords(da, "Y", multidim=True)

    def test_multidim_name_preference_ignores_scalars(self):
        ds = xr.Dataset(
            coords={
                "lat": 10.0,
                "y": ("y", [0.0, 1.0], {"axis": "Y"}),
            }
        )
        xr.testing.assert_identical(get_dim_coords(ds, "Y", multidim=True), ds.y)


class TestGetCoordsByName:
    def test_raises_error_if_coordinate_not_found(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(data=np.ones(3), dims="lat"),
                "lon": xr.DataArray(data=np.ones(3), dims="lon"),
            }
        )

        with pytest.raises(
            KeyError, match="Coordinate with name 'T' not found in the dataset."
        ):
            get_coords_by_name(ds, "T")

    def test_returns_coordinate_from_dataset(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(data=np.ones(3), dims="lat"),
                "lon": xr.DataArray(data=np.ones(3), dims="lon"),
                "time": xr.DataArray(data=np.arange(3), dims="time"),
            }
        )

        coord = get_coords_by_name(ds, "T")
        assert coord.identical(ds["time"])

    def test_returns_coordinate_from_dataarray(self):
        da = xr.DataArray(
            data=np.random.rand(3, 3),
            dims=["lat", "lon"],
            coords={
                "lat": xr.DataArray(data=np.arange(3), dims="lat"),
                "lon": xr.DataArray(data=np.arange(3), dims="lon"),
            },
        )

        coord = get_coords_by_name(da, "X")
        assert coord.identical(da["lon"])

    def test_returns_coordinate_from_curvilinear_dataset(self):
        ds = xr.Dataset(
            coords={
                "nlat": xr.DataArray(data=np.arange(3), dims="nlat"),
                "nlon": xr.DataArray(data=np.arange(3), dims="nlon"),
                "lat": xr.DataArray(
                    data=np.random.rand(3, 3),
                    dims=["nlat", "nlon"],
                    attrs={"standard_name": "latitude"},
                ),
                "lon": xr.DataArray(
                    data=np.random.rand(3, 3),
                    dims=["nlat", "nlon"],
                    attrs={"standard_name": "longitude"},
                ),
            }
        )

        coord = get_coords_by_name(ds, "X")
        assert coord.identical(ds["lon"])

        coord = get_coords_by_name(ds, "Y")
        assert coord.identical(ds["lat"])


class TestCenterTimes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_raises_error_if_time_coord_var_does_not_exist_in_dataset(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")

        with pytest.raises(KeyError):
            center_times(ds)

    def test_skips_centering_time_coords_for_a_dimension_if_bounds_do_not_exist(self):
        ds = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-01-31T12:00:00.000000000",
                            "2000-02-29T12:00:00.000000000",
                            "2000-03-31T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
            },
        )

        # Compare result of the method against the expected.
        result = center_times(ds)
        expected = ds.copy()

        assert result.identical(expected)

    def test_returns_time_coords_as_the_midpoint_between_time_bounds(self):
        ds = xr.Dataset(
            data_vars={
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                "2000-01-01T00:00:00.000000000",
                                "2000-02-01T00:00:00.000000000",
                            ],
                            [
                                "2000-02-01T00:00:00.000000000",
                                "2000-03-01T00:00:00.000000000",
                            ],
                            [
                                "2000-03-01T00:00:00.000000000",
                                "2000-04-01T00:00:00.000000000",
                            ],
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time", "bnds"],
                    attrs={
                        "xcdat_bounds": "True",
                    },
                ),
            },
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-01-31T12:00:00.000000000",
                            "2000-02-29T12:00:00.000000000",
                            "2000-03-31T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
                "time2": xr.DataArray(
                    name="time2",
                    data=np.array(
                        [
                            "2000-01-31T12:00:00.000000000",
                            "2000-02-29T12:00:00.000000000",
                            "2000-03-31T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
            },
        )

        result = center_times(ds)

        # Compare result of the method against the expected.
        expected = xr.Dataset(
            data_vars={
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                "2000-01-01T00:00:00.000000000",
                                "2000-02-01T00:00:00.000000000",
                            ],
                            [
                                "2000-02-01T00:00:00.000000000",
                                "2000-03-01T00:00:00.000000000",
                            ],
                            [
                                "2000-03-01T00:00:00.000000000",
                                "2000-04-01T00:00:00.000000000",
                            ],
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time", "bnds"],
                    attrs={
                        "xcdat_bounds": "True",
                    },
                ),
            },
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-01-16T12:00:00.000000000",
                            "2000-02-15T12:00:00.000000000",
                            "2000-03-16T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
                "time2": xr.DataArray(
                    name="time2",
                    data=np.array(
                        [
                            "2000-01-16T12:00:00.000000000",
                            "2000-02-15T12:00:00.000000000",
                            "2000-03-16T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
            },
        )

        assert result.identical(expected)


class TestSwapLonAxis:
    def test_raises_error_if_no_longitude_axis_exists(self):
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=True)
        ds = ds.drop_dims("lon")

        with pytest.raises(KeyError):
            swap_lon_axis(ds, to=(-180, 180))

    def test_raises_error_with_incorrect_lon_orientation_for_swapping(self):
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=True)
        with pytest.raises(ValueError):
            swap_lon_axis(ds, to=9000)  # type: ignore

    def test_raises_error_if_lon_bounds_contains_more_than_one_prime_meridian_cell(
        self,
    ):
        ds_180 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-180, -1, 0, 1, 179]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                )
            },
            data_vars={
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-180.5, -1.5],
                            [-1.5, -0.5],
                            [-0.5, 0.5],
                            [0.5, 1.5],
                            [-180.5, 1.5],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
            },
        )

        with pytest.raises(ValueError):
            swap_lon_axis(ds_180, to=(0, 360))

    def test_does_not_swap_if_desired_orientation_is_the_same_as_the_existing_orientation(
        self,
    ):
        ds_360 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                )
            },
            data_vars={
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[0, 120], [120, 181], [181, 360]]),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                )
            },
        )

        result = swap_lon_axis(ds_360, to=(0, 360), sort_ascending=False)

        assert result.identical(ds_360)

    def test_does_not_swap_bounds_if_bounds_do_not_exist(self):
        ds_360 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
            },
        )

        result = swap_lon_axis(ds_360, to=(-180, 180))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    data=np.array([-89, 60, 150]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([2, 0, 1]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
            },
        )

        assert result.identical(expected)

    def test_swaps_single_dim_from_360_to_180_and_sorts(self):
        ds_360 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "lon2": xr.DataArray(
                    name="lon2",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[0, 120], [120, 181], [181, 360]]),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )
        result = swap_lon_axis(ds_360, to=(-180, 180))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    data=np.array([-89, 60, 150]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "lon2": xr.DataArray(
                    name="lon2",
                    data=np.array([-89, 60, 150]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([2, 0, 1]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[-179, 0], [0, 120], [120, -179]]),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        assert result.identical(expected)

    def test_swaps_single_dim_from_180_to_360_and_sorts_with_prime_meridian_cell_in_lon_bnds(
        self,
    ):
        ds_180 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-180, -1, 0, 1, 179]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "lon2": xr.DataArray(
                    name="lon2",
                    data=np.array([-180, -1, 0, 1, 179]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-180.5, -1.5],
                            [-1.5, -0.5],
                            [-0.5, 0.5],
                            [0.5, 1.5],
                            [1.5, 179.5],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )
        result = swap_lon_axis(ds_180, to=(0, 360))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0, 1, 179, 180, 359, 360]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "lon2": xr.DataArray(
                    name="lon2",
                    data=np.array([0, 1, 179, 180, 359, 360]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([2, 3, 4, 0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [0, 0.5],
                            [0.5, 1.5],
                            [1.5, 179.5],
                            [179.5, 358.5],
                            [358.5, 359.5],
                            [359.5, 360],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        assert result.identical(expected)

    def test_swaps_all_dims_from_180_to_360_and_sorts_with_prime_meridian_cell_in_lon_bnds(
        self,
    ):
        ds_180 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-180, -1, 0, 1, 179]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "zlon": xr.DataArray(
                    name="zlon",
                    data=np.array([-180, -1, 0, 1, 179]),
                    dims=["zlon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "zlon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "ts2": xr.DataArray(
                    name="ts2",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["zlon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-180.5, -1.5],
                            [-1.5, -0.5],
                            [-0.5, 0.5],
                            [0.5, 1.5],
                            [1.5, 179.5],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
                "zlon_bnds": xr.DataArray(
                    name="zlon_bnds",
                    data=np.array(
                        [
                            [-180.5, -1.5],
                            [-1.5, -0.5],
                            [-0.5, 0.5],
                            [0.5, 1.5],
                            [1.5, 179.5],
                        ]
                    ),
                    dims=["zlon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )
        result = swap_lon_axis(ds_180, to=(0, 360))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0, 1, 179, 180, 359, 360]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "zlon": xr.DataArray(
                    name="zlon",
                    data=np.array([0, 1, 179, 180, 359, 360]),
                    dims=["zlon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "zlon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([2, 3, 4, 0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "ts2": xr.DataArray(
                    name="ts2",
                    data=np.array([2, 3, 4, 0, 1, 2]),
                    dims=["zlon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [0, 0.5],
                            [0.5, 1.5],
                            [1.5, 179.5],
                            [179.5, 358.5],
                            [358.5, 359.5],
                            [359.5, 360],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
                "zlon_bnds": xr.DataArray(
                    name="zlon_bnds",
                    data=np.array(
                        [
                            [0, 0.5],
                            [0.5, 1.5],
                            [1.5, 179.5],
                            [179.5, 358.5],
                            [358.5, 359.5],
                            [359.5, 360],
                        ]
                    ),
                    dims=["zlon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        assert result.identical(expected)


def _make_global_360_dataset(
    seam_values: tuple[float, float] = (0.0, 360.0),
    n: int = 361,
    include_bounds: bool = True,
):
    """Builds a [0, 360] global grid whose two ends share the seam meridian."""
    lon = np.linspace(0.0, 360.0, n)
    data = np.arange(n, dtype=float)
    data[0], data[-1] = seam_values

    coords = {
        "lon": xr.DataArray(
            lon,
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
        )
    }
    data_vars = {
        "ts": xr.DataArray(data, dims=["lon"], attrs={"test_attr": "test"}),
    }

    if include_bounds:
        coords["bnds"] = xr.DataArray(np.array([0, 1]), dims=["bnds"])
        lon_bnds = np.stack(
            [np.clip(lon - 0.5, 0.0, 360.0), np.clip(lon + 0.5, 0.0, 360.0)],
            axis=1,
        )
        data_vars["lon_bnds"] = xr.DataArray(
            lon_bnds, dims=["lon", "bnds"], attrs={"xcdat_bounds": "True"}
        )

    return xr.Dataset(coords=coords, data_vars=data_vars)


def _make_global_180_dataset(
    seam_values: tuple[float, float] = (0.0, 360.0), n: int = 361
):
    """Builds a [-180, 180] global grid whose two ends share the dateline."""
    lon = np.linspace(-180.0, 180.0, n)
    data = np.arange(n, dtype=float)
    data[0], data[-1] = seam_values

    lon_bnds = np.stack(
        [np.clip(lon - 0.5, -180.0, 180.0), np.clip(lon + 0.5, -180.0, 180.0)],
        axis=1,
    )

    return xr.Dataset(
        coords={
            "lon": xr.DataArray(
                lon,
                dims=["lon"],
                attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
            ),
            "bnds": xr.DataArray(np.array([0, 1]), dims=["bnds"]),
        },
        data_vars={
            "ts": xr.DataArray(data, dims=["lon"], attrs={"test_attr": "test"}),
            "lon_bnds": xr.DataArray(
                lon_bnds, dims=["lon", "bnds"], attrs={"xcdat_bounds": "True"}
            ),
        },
    )


def _add_latitude_axis(ds: xr.Dataset) -> xr.Dataset:
    """Adds a two-point latitude axis so 2-D spatial averaging is possible."""
    ds = ds.assign_coords(
        lat=xr.DataArray(
            np.array([-0.5, 0.5]),
            dims=["lat"],
            attrs={"axis": "Y", "bounds": "lat_bnds", "units": "degrees_north"},
        )
    )
    ds["lat_bnds"] = xr.DataArray(
        np.array([[-1.0, 0.0], [0.0, 1.0]]),
        dims=["lat", "bnds"],
        attrs={"xcdat_bounds": "True"},
    )
    ds["ts"] = xr.DataArray(np.tile(ds["ts"].values, (2, 1)), dims=["lat", "lon"])

    return ds


class TestSwapLonAxisSeamDedup:
    def test_raises_error_with_incorrect_seam_side(self):
        ds = _make_global_360_dataset()

        with pytest.raises(ValueError):
            swap_lon_axis(ds, to=(-180, 180), seam_side="middle")  # type: ignore

    def test_raises_error_on_conflicting_seam_values_without_authority(self):
        ds = _make_global_360_dataset(seam_values=(0.0, 360.0))

        with pytest.raises(ValueError, match="Conflicting values"):
            swap_lon_axis(ds, to=(-180, 180))

    def test_retains_single_valid_value_when_other_side_is_missing(self):
        ds = _make_global_360_dataset(seam_values=(np.nan, 360.0))

        result = swap_lon_axis(ds, to=(-180, 180))

        assert result.sizes["lon"] == 360
        assert len(np.unique(result.lon.values)) == 360
        # The collision of 0/360 lands on the 0.0 meridian in this orientation.
        np.testing.assert_allclose(result.ts.sel(lon=0.0).values, 360.0)

    def test_retains_missing_value_when_both_seam_sides_are_missing(self):
        ds = _make_global_360_dataset(seam_values=(np.nan, np.nan))

        result = swap_lon_axis(ds, to=(-180, 180))

        assert result.sizes["lon"] == 360
        assert np.isnan(result.ts.sel(lon=0.0).values)

    def test_seam_side_start_retains_start_side_value(self):
        ds = _make_global_360_dataset(seam_values=(0.0, 360.0))

        result = swap_lon_axis(ds, to=(-180, 180), seam_side="start")

        assert result.sizes["lon"] == 360
        np.testing.assert_allclose(result.ts.sel(lon=0.0).values, 0.0)

    def test_seam_side_end_retains_end_side_value(self):
        ds = _make_global_360_dataset(seam_values=(0.0, 360.0))

        result = swap_lon_axis(ds, to=(-180, 180), seam_side="end")

        assert result.sizes["lon"] == 360
        np.testing.assert_allclose(result.ts.sel(lon=0.0).values, 360.0)

    def test_equal_seam_values_merge_without_authority(self):
        ds = _make_global_360_dataset(seam_values=(7.0, 7.0))

        result = swap_lon_axis(ds, to=(-180, 180))

        assert result.sizes["lon"] == 360
        np.testing.assert_allclose(result.ts.sel(lon=0.0).values, 7.0)

    def test_dedupes_without_bounds(self):
        ds = _make_global_360_dataset(include_bounds=False)

        result = swap_lon_axis(ds, to=(-180, 180), seam_side="start")

        assert result.sizes["lon"] == 360
        assert "lon_bnds" not in result
        np.testing.assert_allclose(result.ts.sel(lon=0.0).values, 0.0)

    def test_dedupes_180_native_grid_on_swap_to_360(self):
        ds = _make_global_180_dataset(seam_values=(0.0, 360.0))

        result = swap_lon_axis(ds, to=(0, 360), seam_side="start")

        # The dateline endpoints map to distinct meridians (0 and 180) on the
        # 360 axis, so no collision occurs and the grid keeps all its points.
        assert result.sizes["lon"] == 361
        assert len(np.unique(result.lon.values)) == 361
        assert "bnds" in result.coords
        np.testing.assert_allclose(
            result.spatial.get_weights(axis=["X"]).sum().values, 360.0
        )

    def test_merges_bounds_so_support_area_is_counted_once(self):
        ds = _make_global_360_dataset(seam_values=(0.0, 360.0))

        result = swap_lon_axis(ds, to=(-180, 180), seam_side="start")
        weights = result.spatial.get_weights(axis=["X"])

        np.testing.assert_allclose(weights.sum().values, 360.0)

    def test_spatial_mean_stable_across_orientations_with_equal_seam(self):
        ds = _add_latitude_axis(
            _make_global_360_dataset(seam_values=(7.0, 7.0))
        )

        mean_orig = ds.spatial.average("ts")["ts"].values
        result_180 = swap_lon_axis(ds, to=(-180, 180))
        result_360 = swap_lon_axis(result_180, to=(0, 360))

        np.testing.assert_allclose(
            result_180.spatial.average("ts")["ts"].values, mean_orig
        )
        np.testing.assert_allclose(
            result_360.spatial.average("ts")["ts"].values, mean_orig
        )

    def test_round_trip_swap_preserves_grid_identity_area_and_mean(self):
        ds = _add_latitude_axis(
            _make_global_360_dataset(seam_values=(0.0, 360.0))
        )

        first = swap_lon_axis(ds, to=(-180, 180), seam_side="start")
        second = swap_lon_axis(first, to=(0, 360), seam_side="start")
        third = swap_lon_axis(second, to=(-180, 180), seam_side="start")
        fourth = swap_lon_axis(third, to=(0, 360), seam_side="start")

        # Re-visiting an orientation reproduces it exactly. The seam cell is a
        # single out-of-order cell on the (-180, 180) axis but two split cells
        # on the (0, 360) axis, so each orientation keeps its own fixed size.
        xr.testing.assert_identical(first, third)
        xr.testing.assert_identical(second, fourth)
        assert first.sizes["lon"] == third.sizes["lon"]
        assert second.sizes["lon"] == fourth.sizes["lon"]

        for result in (first, second, third, fourth):
            np.testing.assert_allclose(
                result.spatial.get_weights(axis=["X"]).sum().values, 360.0
            )

        np.testing.assert_allclose(
            first.spatial.average("ts")["ts"].values,
            third.spatial.average("ts")["ts"].values,
        )
        np.testing.assert_allclose(
            second.spatial.average("ts")["ts"].values,
            fourth.spatial.average("ts")["ts"].values,
        )
        np.testing.assert_allclose(
            first.spatial.average("ts")["ts"].values,
            second.spatial.average("ts")["ts"].values,
        )

    def test_round_trip_through_netcdf_write_and_reread_is_stable(self, tmp_path):
        ds = _make_global_360_dataset(seam_values=(0.0, 360.0))
        first = swap_lon_axis(ds, to=(-180, 180), seam_side="start")
        second = swap_lon_axis(first, to=(0, 360), seam_side="start")

        filepath = tmp_path / "seam_round_trip.nc"
        first.to_netcdf(filepath)
        reread = xr.open_dataset(filepath).load()

        assert reread.sizes["lon"] == 360
        reread_swapped = swap_lon_axis(reread, to=(0, 360), seam_side="start")
        xr.testing.assert_identical(reread_swapped, second)

    def test_dedup_synchronizes_mask_and_data_variable(self):
        ds = _make_global_360_dataset(seam_values=(0.0, 360.0))
        ds["mask"] = xr.DataArray(
            np.tile(np.arange(361) % 2 == 0, 1), dims=["lon"]
        )

        result = swap_lon_axis(ds, to=(-180, 180), seam_side="start")

        assert result.sizes["lon"] == 360
        assert result.ts.sizes["lon"] == 360
        assert result.mask.sizes["lon"] == 360
        assert result.lon_bnds.sizes["lon"] == 360
        assert result.mask.dtype == bool

    def test_dedup_leaves_non_longitude_variables_untouched(self):
        ds = _make_global_360_dataset(seam_values=(0.0, 360.0))
        ds = ds.assign_coords(
            lat=xr.DataArray(
                np.array([-0.5, 0.5]),
                dims=["lat"],
                attrs={"axis": "Y", "bounds": "lat_bnds"},
            )
        )
        ds["lat_bnds"] = xr.DataArray(
            np.array([[-1.0, 0.0], [0.0, 1.0]]),
            dims=["lat", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["field"] = xr.DataArray(
            np.zeros((2, 361)), dims=["lat", "lon"]
        )

        result = swap_lon_axis(ds, to=(-180, 180), seam_side="start")

        assert result.lat_bnds.dims == ("lat", "bnds")
        assert result.sizes["lat"] == 2
        assert result.field.sizes == {"lat": 2, "lon": 360}

    def test_dedup_supports_dask_backed_variables(self):
        import dask.array as da

        ds = _make_global_360_dataset(seam_values=(0.0, 360.0))
        ds["ts"] = xr.DataArray(
            da.from_array(np.arange(361.0), chunks=100), dims=["lon"]
        )

        result = swap_lon_axis(ds, to=(-180, 180), seam_side="end")

        assert result.ts.chunks is not None
        np.testing.assert_allclose(result.ts.sel(lon=0.0).values, 360.0)
        np.testing.assert_allclose(
            result.ts.values,
            swap_lon_axis(
                ds.load(), to=(-180, 180), seam_side="end"
            ).ts.values,
        )

    def test_non_global_grid_is_not_deduped(self):
        ds = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    np.arange(0.0, 359.0, 1.0),
                    dims=["lon"],
                    attrs={"axis": "X", "units": "degrees_east"},
                )
            },
            data_vars={"ts": xr.DataArray(np.arange(359.0), dims=["lon"])},
        )

        result = swap_lon_axis(ds, to=(-180, 180))

        assert result.sizes["lon"] == 359

    def test_raises_error_when_more_than_one_seam_pair_collides(self):
        # Two points already sit on 0, plus the 360 endpoint also wraps to 0.
        ds = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    np.array([0.0, 0.0, 90.0, 180.0, 270.0, 360.0]),
                    dims=["lon"],
                    attrs={"axis": "X", "units": "degrees_east"},
                )
            },
            data_vars={"ts": xr.DataArray(np.arange(6.0), dims=["lon"])},
        )

        with pytest.raises(ValueError, match="single pair of duplicate"):
            swap_lon_axis(ds, to=(-180, 180), seam_side="start")
