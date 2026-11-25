"""
Axis module for utilities related to axes, including functions to manipulate
coordinates.
"""

from typing import Literal

import cf_xarray as cfxr  # noqa: F401
import numpy as np
import xarray as xr

from xcdat.utils import _if_multidim_dask_array_then_load

# https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axis-names
CFAxisKey = Literal["X", "Y", "T", "Z"]
# Which side of the original longitude axis is authoritative when the two
# ends of the axis map to the same meridian (the seam) after a swap.
SeamSide = Literal["start", "end"]
# https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#coordinate-names
CFStandardNameKey = Literal[
    "latitude", "longitude", "time", "vertical", "height", "pressure"
]

# A dictionary that maps the xCDAT `axis` arguments to keys used for `cf_xarray`
# accessor class indexing. For example, if we pass `axis="X"` to a function,
# we can fetch specific `cf_xarray` mapping tables such as `ds.cf.axes["X"]`
# or `ds.cf.coordinates["longitude"]`.
# More information: https://cf-xarray.readthedocs.io/en/latest/coord_axes.html
CF_ATTR_MAP: dict[CFAxisKey, dict[str, CFAxisKey | CFStandardNameKey]] = {
    "X": {"axis": "X", "coordinate": "longitude"},
    "Y": {"axis": "Y", "coordinate": "latitude"},
    "T": {"axis": "T", "coordinate": "time"},
    "Z": {"axis": "Z", "coordinate": "vertical"},
}

COORD_DEFAULT_ATTRS: dict[CFAxisKey, dict[str, str | CFAxisKey | CFStandardNameKey]] = {
    "X": dict(units="degrees_east", **CF_ATTR_MAP["X"]),
    "Y": dict(units="degrees_north", **CF_ATTR_MAP["Y"]),
    "T": dict(calendar="standard", **CF_ATTR_MAP["T"]),
    "Z": dict(**CF_ATTR_MAP["Z"]),
}

# A dictionary that maps common variable names to coordinate variables. This
# map is used as fall-back when coordinate variables don't have CF attributes
# set for ``cf_xarray`` to interpret using `CF_ATTR_MAP`.
VAR_NAME_MAP: dict[CFAxisKey, list[str]] = {
    "X": ["longitude", "lon"],
    "Y": ["latitude", "lat"],
    "T": ["time"],
    "Z": ["vertical", "height", "pressure", "lev", "plev"],
}


def get_dim_keys(obj: xr.Dataset | xr.DataArray, axis: CFAxisKey) -> str | list[str]:
    """Gets the dimension key(s) for an axis.

    Each dimension should have a corresponding dimension coordinate variable,
    which has a 1:1 map in keys and is denoted by the * symbol when printing out
    the xarray object.


    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The Dataset or DataArray object.
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", or "Z")

    Returns
    -------
    str | list[str]
        The dimension string or a list of dimensions strings for an axis.
    """
    dims = sorted([str(dim) for dim in get_dim_coords(obj, axis).dims])

    return dims[0] if len(dims) == 1 else dims


def get_dim_coords(
    obj: xr.Dataset | xr.DataArray, axis: CFAxisKey, multidim: bool = False
) -> xr.Dataset | xr.DataArray:
    """Gets the dimension coordinates for an axis.

    This function uses ``cf_xarray`` to attempt to map the axis to its
    dimension coordinates by interpreting the CF axis and coordinate names
    found in the coordinate attributes. Refer to [1]_ for a list of CF axis and
    coordinate names that can be interpreted by ``cf_xarray``.

    If ``obj`` is an ``xr.Dataset,``, this function can return a single
    dimension coordinate variable as an ``xr.DataArray`` or multiple dimension
    coordinate variables in an ``xr Dataset``. If ``obj`` is an ``xr.DataArray``,
    this function should return a single dimension coordinate variable as an
    ``xr.DataArray``.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The Dataset or DataArray object.
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", "Z").
    multidim : bool
        If True, also consider non-index coordinates, including multidimensional
        coordinates. Scalar coordinates are excluded. Defaults to False.

    Returns
    -------
    xr.Dataset | xr.DataArray
        A Dataset of dimension coordinate variables or a DataArray for
        the single dimension coordinate variable.

    Raises
    ------
    ValueError
        If the ``obj`` is an ``xr.DataArray`` and more than one dimension is
        mapped to the same axis.
    KeyError
        If no dimension coordinate variables were found for the ``axis``.

    Notes
    -----
    Coordinates are identified using CF metadata or recognized common names.
    Multidimensional coordinates are ignored unless ``multidim=True``.
    In multidimensional mode, CF coordinate mappings take precedence over
    axis-only mappings and common names (e.g., 2D latitude over a grid index).
    If no CF coordinate mapping exists for a horizontal axis, recognized
    latitude/longitude names take precedence over axis-only mappings.

    References
    ----------
    .. [1] https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axes-and-coordinates
    """
    if multidim:
        # Multidimensional coordinates cannot be indexes; exclude only scalars.
        candidate_keys = [name for name, coord in obj.coords.items() if coord.ndim > 0]

        cf_names = obj.cf.coordinates.get(CF_ATTR_MAP[axis]["coordinate"], [])
        cf_keys = [name for name in candidate_keys if name in cf_names]
        geographic_keys = (
            [name for name in candidate_keys if name in VAR_NAME_MAP[axis]]
            if axis in ("X", "Y")
            else []
        )

        # Prefer CF physical coordinates, then geographic names, over grid indexes.
        candidate_keys = cf_keys or geographic_keys or candidate_keys
    else:
        # Get the object's index keys, with each being a dimension.
        # NOTE: xarray does not include multidimensional coordinates as index keys.
        # Example: ["lat", "lon", "time"]
        candidate_keys = list(obj.indexes.keys())

    # Identify the requested axis using CF metadata and recognized common names.
    coord_keys = _get_all_coord_keys(obj, axis)
    dim_coord_keys = list(set(candidate_keys) & set(coord_keys))

    if isinstance(obj, xr.DataArray) and len(dim_coord_keys) > 1:
        raise ValueError(
            f"This DataArray has more than one dimension {dim_coord_keys} mapped to the "
            f"'{axis}' axis, which is an unexpected behavior. Try dropping extraneous "
            "dimensions from the DataArray first (might affect data shape)."
        )

    if len(dim_coord_keys) == 0:
        raise KeyError(
            f"No '{axis}' axis dimension coordinate variables were found in the "
            f"xarray object. Make sure dimension coordinate variables exist, they are "
            "one dimensional, and their CF 'axis' or 'standard_name' attrs are "
            "correctly set."
        )

    dim_coords = obj[
        dim_coord_keys if len(dim_coord_keys) > 1 else dim_coord_keys[0]
    ].copy()

    return dim_coords


def get_coords_by_name(obj: xr.Dataset | xr.DataArray, axis: CFAxisKey) -> xr.DataArray:
    """Retrieve the coordinate variable based on its name.

    This method is useful for returning the desired coordinate in the following
    cases:

    - Coordinates that are not CF-compliant (e.g., missing CF attributes like
      "axis", "standard_name", or "long_name") but use common names.
    - Axes with multiple sets of coordinates. For example, curvilinear grids may
      have multiple coordinates for the same axis (e.g., (nlat, lat) for X and
      (nlon, lon) for Y). In most cases, "lat" and "lon" are the desired
      coordinates, which this function will return.

    Common variable names for each axis (from ``VAR_NAME_MAP``):

    - "X" axis: ["longitude", "lon"]
    - "Y" axis: ["latitude", "lat"]
    - "T" axis: ["time"]
    - "Z" axis: ["vertical", "height", "pressure", "lev", "plev"]

    Parameters
    ----------
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", or "Z").

    Returns
    -------
    xr.DataArray
        The coordinate variable.

    Raises
    ------
    KeyError
        If the coordinate variable is not found in the dataset.
    """
    coord_names = VAR_NAME_MAP[axis]

    for coord_name in coord_names:
        if coord_name in obj.coords:
            coord = obj.coords[coord_name]

            return coord

    raise KeyError(f"Coordinate with name '{axis}' not found in the dataset.")


def center_times(dataset: xr.Dataset) -> xr.Dataset:
    """Centers time coordinates using the midpoint between time bounds.

    Time coordinates can be recorded using different intervals, including
    the beginning, middle, or end of the interval. Centering time
    coordinates, ensures calculations using these values are performed
    reliably regardless of the recorded interval.

    This method attempts to get bounds for each time variable using the
    CF "bounds" attribute. Coordinate variables that cannot be mapped to
    bounds will be skipped.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset with original time coordinates.

    Returns
    -------
    xr.Dataset
        The Dataset with centered time coordinates.
    """
    ds = dataset.copy()
    coords = get_dim_coords(ds, "T")

    for coord in coords.coords.values():
        try:
            coord_bounds = ds.bounds.get_bounds("T", str(coord.name))
        except KeyError:
            coord_bounds = None

        if coord_bounds is not None:
            lower_bounds, upper_bounds = (
                coord_bounds[:, 0].data,
                coord_bounds[:, 1].data,
            )
            bounds_diffs: np.timedelta64 = (upper_bounds - lower_bounds) / 2
            bounds_mids: np.ndarray = lower_bounds + bounds_diffs

            coord_centered = xr.DataArray(
                name=coord.name,
                data=bounds_mids,
                dims=coord.dims,
                attrs=coord.attrs,
            )
            coord_centered.encoding = coord.encoding
            ds = ds.assign_coords({coord.name: coord_centered})

            # Update time bounds with centered time coordinates.
            coord_bounds[coord_centered.name] = coord_centered
            ds[coord_bounds.name] = coord_bounds

    return ds


def swap_lon_axis(
    dataset: xr.Dataset,
    to: tuple[float, float],
    sort_ascending: bool = True,
    seam_side: SeamSide | None = None,
) -> xr.Dataset:
    """Swaps the orientation of a dataset's longitude axis.

    This method also swaps the axis orientation of the longitude bounds if it
    exists. Afterwards, it sorts longitude and longitude bounds values in
    ascending order.

    When the axis covers the full globe, its two ends represent the same
    meridian (the seam, e.g. 0/360 or -180/180). After a swap these two grid
    points collide on the same longitude. The collision is resolved by merging
    the two points into a single grid point:

    * If ``seam_side`` is specified, the value from that side of the original
      axis (``"start"`` or ``"end"``) is treated as authoritative.
    * If ``seam_side`` is None and the two values conflict (both are valid but
      differ), a ``ValueError`` is raised. If only one side is valid (the other
      is missing/NaN), the valid value is retained.

    Bounds are merged alongside the data so the support area of the seam cell
    is counted exactly once.

    Note, based on how datasets are chunked, swapping the longitude dimension
    and sorting might raise ``PerformanceWarning: Slicing is producing a
    large chunk. To accept the large chunk and silence this warning, set the
    option...``. This function uses xarray's arithmetic to swap orientations,
    so this warning seems potentially unavoidable.

    Parameters
    ----------
    dataset : xr.Dataset
         The Dataset containing a longitude axis.
    to : tuple[float, float]
        The orientation to swap the Dataset's longitude axis to. Supported
        orientations include:

        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation
    sort_ascending : bool
        After swapping, sort in ascending order (True), or keep existing order
        (False).
    seam_side : {"start", "end"} | None, optional
        Which side of the original (pre-swap) longitude axis is authoritative
        for the seam cell whose two ends collide after the swap:

        * "start": retain the value at the start of the original axis.
        * "end": retain the value at the end of the original axis.
        * None (default): retain the single valid side if one side is missing,
          but raise a ``ValueError`` if both sides hold differing valid values.

    Returns
    -------
    xr.Dataset
        The Dataset with swapped lon axes orientation.

    Raises
    ------
    ValueError
        If the two sides of the seam hold conflicting valid values and
        ``seam_side`` is not specified, or if ``seam_side`` is not one of
        "start", "end", or None.
    """
    if seam_side is not None and seam_side not in ("start", "end"):
        raise ValueError(
            f"Incorrect `seam_side` argument value '{seam_side}'. Supported "
            "values include: 'start', 'end', or None."
        )

    ds = dataset.copy()
    coords = get_dim_coords(ds, "X").coords
    coord_keys = list(coords.keys())

    # Attempt to swap the orientation for longitude coordinates.
    for key in coord_keys:
        new_coord = _swap_lon_axis(ds.coords[key], to)

        if ds.coords[key].identical(new_coord):
            continue

        ds.coords[key] = new_coord

    try:
        bounds = ds.bounds.get_bounds("X")
    except KeyError:
        bounds = None

    if isinstance(bounds, xr.DataArray):
        ds = _swap_lon_bounds(ds, str(bounds.name), to)
    elif isinstance(bounds, xr.Dataset):
        for key in bounds.data_vars.keys():
            ds = _swap_lon_bounds(ds, str(key), to)

    # The two ends of a global longitude axis represent the same meridian and
    # collide after a swap. Merge the duplicate seam points (data, masks, and
    # bounds) so the seam cell exists exactly once.
    ds = _dedupe_lon_seam(ds, seam_side)

    if sort_ascending:
        ds = ds.sortby(list(coords.dims), ascending=True)

    return ds


def _dedupe_lon_seam(ds: xr.Dataset, seam_side: SeamSide | None) -> xr.Dataset:
    """Merges the duplicate grid points that collide on the longitude seam.

    The two ends of a global longitude axis represent the same meridian (e.g.
    0/360 or -180/180) and map to the same coordinate value after an axis
    orientation swap. This function detects that collision pair for each
    longitude dimension and merges it into a single grid point across every
    variable, coordinate, and bounds attached to the dimension.

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset after swapping longitude orientation (not yet sorted).
    seam_side : {"start", "end"} | None
        Which side of the original axis is authoritative. If None, a conflict
        between two valid values raises a ``ValueError``; a missing value on
        one side falls back to the valid side.

    Returns
    -------
    xr.Dataset
        The Dataset with each longitude seam collision merged into one point.

    Raises
    ------
    ValueError
        If a longitude dimension has a seam collision that is not a single
        pair, or if conflicting valid values exist and ``seam_side`` is None.
    """
    x_dims = get_dim_keys(ds, "X")
    x_dims = [x_dims] if isinstance(x_dims, str) else x_dims

    # Snapshot auxiliary coordinates (e.g. "bnds") so they can be restored
    # after concatenating the merged seam point back into the dataset.
    original_coords = ds.coords.copy()

    try:
        bounds = ds.bounds.get_bounds("X")
    except KeyError:
        bounds = None

    if isinstance(bounds, xr.DataArray):
        bounds_keys = [str(bounds.name)]
    elif isinstance(bounds, xr.Dataset):
        bounds_keys = [str(key) for key in bounds.data_vars.keys()]
    else:
        bounds_keys = []

    for dim in x_dims:
        # Indices are still in the original order at this stage, so the first
        # collision index is the "start" side and the second the "end" side.
        _, inverse, counts = np.unique(
            ds.coords[dim].values, return_inverse=True, return_counts=True
        )

        if not np.any(counts > 1):
            continue

        collision_values = np.where(counts > 1)[0]
        if collision_values.size != 1 or counts[collision_values[0]] != 2:
            duplicate_lons = ds.coords[dim].values[
                np.isin(inverse, collision_values)
            ]
            raise ValueError(
                f"Expected a single pair of duplicate longitude points on the "
                f"'{dim}' seam, but found duplicate longitude values "
                f"{duplicate_lons.tolist()}."
            )

        start_idx, end_idx = np.where(inverse == collision_values[0])[0]
        ds = _merge_lon_seam_pair(
            ds,
            original_coords,
            x_dims,
            dim,
            int(start_idx),
            int(end_idx),
            seam_side,
            bounds_keys,
        )

    return ds


def _merge_lon_seam_pair(
    ds: xr.Dataset,
    original_coords: xr.Coordinates,
    x_dims: list[str],
    dim: str,
    start_idx: int,
    end_idx: int,
    seam_side: SeamSide | None,
    bounds_keys: list[str],
) -> xr.Dataset:
    """Merges a single duplicate seam pair on ``dim`` into one grid point.

    Only variables and coordinates that use ``dim`` are modified; unrelated
    variables (e.g. latitude bounds) are passed through untouched.

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset containing the duplicate seam pair.
    original_coords : xr.Coordinates
        Snapshot of the dataset coordinates before the merge, used to restore
        auxiliary dimension coordinates (e.g. the "bnds" dimension).
    x_dims : list[str]
        All longitude dimension names, used to skip coordinates belonging to
        another (not yet merged) longitude dimension.
    dim : str
        The longitude dimension name.
    start_idx : int
        Index of the point from the start side of the original axis.
    end_idx : int
        Index of the point from the end side of the original axis.
    seam_side : {"start", "end"} | None
        Authoritative side when both values are valid but differ.
    bounds_keys : list[str]
        Names of longitude bounds variables, which are merged by union rather
        than value selection so the support area is counted once.

    Returns
    -------
    xr.Dataset
        The Dataset with the end-side duplicate point removed.
    """
    start_side = ds.isel({dim: start_idx})
    end_side = ds.isel({dim: end_idx})

    lon_names = [
        str(name)
        for name in list(ds.data_vars) + list(ds.coords)
        if dim in ds[name].dims
    ]

    # Validate that every non-bounds variable can resolve its seam pair.
    _validate_seam_pairs(
        lon_names, bounds_keys, start_side, end_side, seam_side, dim
    )

    # Build the merged seam point for each variable on the axis.
    seam_vars = _build_merged_seam_vars(
        lon_names, bounds_keys, start_side, end_side, seam_side, dim
    )

    # Rebuild the longitude-axis variables with the single merged seam point.
    keep_indices = [
        i for i in range(ds.sizes[dim]) if i not in (start_idx, end_idx)
    ]
    position = sum(i < start_idx for i in keep_indices)
    ds_dropped = ds.isel({dim: keep_indices})

    rebuilt = {
        name: xr.concat(
            (
                ds_dropped[name].isel({dim: slice(0, position)}),
                seam_vars[name].expand_dims(dim),
                ds_dropped[name].isel({dim: slice(position, None)}),
            ),
            dim=dim,
        )
        for name in lon_names
    }

    # Recombine rebuilt longitude variables with untouched non-longitude vars.
    ds_no_lon = ds.get(
        [v for v in ds.data_vars if dim not in ds[v].dims]  # type: ignore
    )
    ds = xr.merge((ds_no_lon, xr.Dataset(rebuilt)), compat="override")
    ds = _restore_auxiliary_coords(ds, original_coords, x_dims, dim)

    return ds


def _validate_seam_pairs(
    lon_names: list[str],
    bounds_keys: list[str],
    start_side: xr.Dataset,
    end_side: xr.Dataset,
    seam_side: SeamSide | None,
    dim: str,
) -> None:
    """Raises if any non-bounds variable has an unresolvable seam conflict.

    Parameters
    ----------
    lon_names : list[str]
        Names of all variables and coordinates on the longitude dimension.
    bounds_keys : list[str]
        Names of longitude bounds variables, which are skipped.
    start_side, end_side : xr.Dataset
        Datasets reduced to the start- and end-side seam points.
    seam_side : {"start", "end"} | None
        Authoritative side for valid-value conflicts.
    dim : str
        The longitude dimension name (for error messages).
    """
    for name in lon_names:
        if name in bounds_keys:
            continue

        _resolve_seam_values(
            start_side[name],
            end_side[name],
            seam_side,
            var_name=name,
            dim=dim,
        )


def _build_merged_seam_vars(
    lon_names: list[str],
    bounds_keys: list[str],
    start_side: xr.Dataset,
    end_side: xr.Dataset,
    seam_side: SeamSide | None,
    dim: str,
) -> dict[str, xr.DataArray]:
    """Builds the merged seam value for each longitude-axis variable.

    Parameters
    ----------
    lon_names : list[str]
        Names of all variables and coordinates on the longitude dimension.
    bounds_keys : list[str]
        Names of longitude bounds variables, which are merged by union.
    start_side, end_side : xr.Dataset
        Datasets reduced to the start- and end-side seam points.
    seam_side : {"start", "end"} | None
        Authoritative side for valid-value conflicts.
    dim : str
        The longitude dimension name.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping of variable name to its merged seam point (with the longitude
        dimension reduced).
    """
    seam_vars: dict[str, xr.DataArray] = {}

    for name in lon_names:
        if name in bounds_keys:
            seam_vars[name] = _merge_seam_bounds(
                start_side[name], end_side[name]
            )
        else:
            seam_vars[name] = _resolve_seam_values(
                start_side[name],
                end_side[name],
                seam_side,
                var_name=name,
                dim=dim,
            )

    return seam_vars


def _restore_auxiliary_coords(
    ds: xr.Dataset,
    original_coords: xr.Coordinates,
    x_dims: list[str],
    dim: str,
) -> xr.Dataset:
    """Restores non-longitude auxiliary coordinates after rebuilding the axis.

    Parameters
    ----------
    ds : xr.Dataset
        The rebuilt dataset.
    original_coords : xr.Coordinates
        Coordinates snapshot taken before the merge.
    x_dims : list[str]
        All longitude dimension names; coordinates on these are skipped since
        they are handled by their own merge iteration.
    dim : str
        The longitude dimension being merged.

    Returns
    -------
    xr.Dataset
        The dataset with auxiliary dimension coordinates (e.g. "bnds")
        restored.
    """
    other_x_dims = [x_dim for x_dim in x_dims if x_dim != dim]
    non_longitude_coords = xr.Coordinates(
        {
            name: coord
            for name, coord in original_coords.items()
            if not any(x_dim in coord.dims for x_dim in other_x_dims)
        }
    )

    return _restore_non_axis_coords(ds, non_longitude_coords, dim)


def _resolve_seam_values(
    start_val: xr.DataArray,
    end_val: xr.DataArray,
    seam_side: SeamSide | None,
    var_name: str,
    dim: str,
) -> xr.DataArray:
    """Resolves a seam value pair for a single (non-bounds) variable.

    Missing values (NaN/NaT) fall back to the valid side regardless of
    ``seam_side``. When both sides are valid but differ, ``seam_side`` decides
    the authoritative value, otherwise a ``ValueError`` is raised.

    Parameters
    ----------
    start_val, end_val : xr.DataArray
        The variable values at the start- and end-side seam points, with the
        seam dimension already reduced.
    seam_side : {"start", "end"} | None
        Authoritative side for valid-value conflicts.
    var_name : str
        Name of the variable (for error messages).
    dim : str
        Name of the longitude dimension (for error messages).

    Returns
    -------
    xr.DataArray
        The resolved seam value (same shape as the inputs minus ``dim``).

    Raises
    ------
    ValueError
        If both values are valid, differ, and ``seam_side`` is None.
    """
    start_missing = start_val.isnull()
    end_missing = end_val.isnull()

    if seam_side == "start":
        return xr.where(start_missing, end_val, start_val, keep_attrs=True)
    if seam_side == "end":
        return xr.where(end_missing, start_val, end_val, keep_attrs=True)

    both_valid = ~(start_missing | end_missing)
    conflict = both_valid & (start_val != end_val)
    # NaN != NaN is False, but equality checks above already exclude missing
    # values through `both_valid`.
    if bool(conflict.any()):
        raise ValueError(
            f"Conflicting values found on the '{dim}' seam for variable "
            f"'{var_name}': the two ends of the longitude axis map to the "
            "same meridian but hold different valid values. Specify "
            "`seam_side='start'` or `seam_side='end'` to declare which side "
            "of the original axis is authoritative."
        )

    return xr.where(start_missing, end_val, start_val, keep_attrs=True)


def _merge_seam_bounds(
    start_bounds: xr.DataArray,
    end_bounds: xr.DataArray,
) -> xr.DataArray:
    """Merges the bounds of a duplicate seam pair into a single cell.

    If the two cells abut within the axis orientation, their shared edge is
    used directly. Otherwise the cell spans the seam (e.g. [359.5, 360] and
    [0, 0.5] -> [359.5, 0.5]), which xCDAT represents with out-of-order bounds
    and handles correctly when generating longitude weights. This ensures the
    merged cell's support area is the union and is counted exactly once.

    Parameters
    ----------
    start_bounds, end_bounds : xr.DataArray
        The 1-D bounds variables (with a trailing bounds dimension) at the
        start- and end-side seam points.

    Returns
    -------
    xr.DataArray
        The merged bounds for the single seam cell.
    """
    s_lo, s_hi = start_bounds.values[0], start_bounds.values[1]
    e_lo, e_hi = end_bounds.values[0], end_bounds.values[1]

    if np.isclose(s_hi, e_lo):
        lo, hi = s_lo, e_hi
    elif np.isclose(e_hi, s_lo):
        lo, hi = e_lo, s_hi
    else:
        # The cell wraps across the seam: keep out-of-order bounds.
        lo, hi = e_lo, s_hi

    merged = start_bounds.copy()
    merged.values = np.array([lo, hi])

    return merged


def _get_all_coord_keys(
    obj: xr.Dataset | xr.DataArray, axis: CFAxisKey
) -> list[str]:
    """Gets all dimension and non-dimension coordinate keys for an axis.

    This function uses ``cf_xarray`` to interpret CF axis and coordinate name
    metadata to map an ``axis`` to its coordinate keys. Refer to [2]_ for more
    information on the ``cf_xarray`` mapping tables.

    It also loops over a list of statically defined coordinate variable names to
    see if they exist in the object, and appends keys that do exist.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The Dataset or DataArray object.
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", or "Z").

    Returns
    -------
    list[str]
        The axis coordinate variable keys.

    References
    ----------
    .. [2] https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axes-and-coordinates
    """
    cf_attrs = CF_ATTR_MAP[axis]
    var_names = VAR_NAME_MAP[axis]

    keys: list[str] = []

    try:
        keys = keys + obj.cf.axes[cf_attrs["axis"]]
    except KeyError:
        pass

    try:
        keys = keys + obj.cf.coordinates[cf_attrs["coordinate"]]
    except KeyError:
        pass

    for name in var_names:
        if name in obj.coords.keys():
            keys.append(name)

    return list(set(keys))


def _swap_lon_bounds(ds: xr.Dataset, key: str, to: tuple[float, float]):
    bounds = ds[key].copy()
    new_bounds = _swap_lon_axis(bounds, to)

    if not ds[key].identical(new_bounds):
        ds[key] = new_bounds

        # Handle cases where a prime meridian cell exists, which can occur
        # after swapping longitude bounds to (0, 360). This involves extending
        # the longitude and bounds by one cell to take into account the prime
        # meridian. It also results in extending the data variables by one
        # value.
        if to == (0, 360):
            p_meridian_index = _get_prime_meridian_index(ds[key])

            if p_meridian_index is not None:
                # A seam cell (the merged cell at the -180/180 dateline) also
                # has out-of-order bounds, but its center is the dateline and
                # not the prime meridian. Only genuine prime-meridian cells are
                # split, otherwise round-trip swaps grow the grid repeatedly.
                center = ds.coords[get_dim_keys(ds[key], "X")][
                    p_meridian_index
                ]
                if not _is_on_dateline(center):
                    ds = _align_lon_to_360(ds, ds[key], p_meridian_index)

    return ds


def _is_on_dateline(lon: xr.DataArray) -> bool:
    """Whether a longitude coordinate sits on the -180/180 dateline seam.

    Parameters
    ----------
    lon : xr.DataArray
        Scalar longitude value(s) in either the (-180, 180) or (0, 360)
        orientation.

    Returns
    -------
    bool
        True if the longitude is equivalent to 180 degrees (i.e. -180 or 180).
    """
    return bool(np.all(np.isclose(np.abs(np.asarray(lon, dtype=float)), 180.0)))


def _swap_lon_axis(coords: xr.DataArray, to: tuple[float, float]) -> xr.DataArray:
    """Swaps the axis orientation for longitude coordinates.

    Parameters
    ----------
    coords : xr.DataArray
        Coordinates on a longitude axis.
    to : tuple[float, float]
        The new longitude axis orientation.

    Returns
    -------
    xr.DataArray
        The longitude coordinates the opposite axis orientation If the
        coordinates are already on the specified axis orientation, the same
        coordinates are returned.
    """
    with xr.set_options(keep_attrs=True):
        if to == (-180, 180):
            # FIXME: Performance warning produced after swapping and then sorting
            # based on how datasets are chunked.
            new_coords = ((coords + 180) % 360) - 180
        elif to == (0, 360):
            # Example with 180 coords: [-180, -0, 179] -> [0, 180, 360]
            # Example with 360 coords: [60, 150, 360] -> [60, 150, 0]
            # FIXME: Performance warning produced after swapping and then sorting
            # based on how datasets are chunked.
            new_coords = coords % 360

            # Check if the original coordinates contain an element with a value
            # of 360. If this element exists, use its index to revert its
            # swapped value of 0 (360 % 360 is 0) back to 360. This case usually
            # happens if the coordinate are already on the (0, 360) axis
            # orientation.
            # Example with 360 coords: [60, 150, 0] -> [60, 150, 360]
            index_with_360 = np.where(coords == 360)

            if index_with_360[0].size > 0:
                _if_multidim_dask_array_then_load(new_coords)

                new_coords[index_with_360] = 360
        else:
            raise ValueError(
                "Currently, only (-180, 180) and (0, 360) are supported longitude axis "
                "orientations."
            )

    new_coords.encoding = coords.encoding

    return new_coords


def _get_prime_meridian_index(lon_bounds: xr.DataArray) -> np.ndarray | None:
    """Gets the index of the prime meridian cell in the longitude bounds.

    A prime meridian cell can exist when converting the axis orientation
    from [-180, 180) to [0, 360).

    Parameters
    ----------
    lon_bounds : xr.DataArray
        The longitude bounds.

    Returns
    -------
    np.ndarray | None
        An array with a single element representing the index of the prime
        meridian index if it exists. Otherwise, None if the cell does not exist.

    Raises
    ------
    ValueError
        If more than one grid cell spans the prime meridian.
    """
    p_meridian_index = np.where(lon_bounds[:, 1] - lon_bounds[:, 0] < 0)[0]

    if p_meridian_index.size == 0:  # pragma:no cover
        return None
    elif p_meridian_index.size > 1:
        raise ValueError("More than one grid cell spans prime meridian.")

    return p_meridian_index


def _align_lon_to_360(
    ds: xr.Dataset,
    lon_bounds: xr.DataArray,
    p_meridian_index: np.ndarray,
) -> xr.Dataset:
    """Handles a prime meridian cell to align longitude axis to (0, 360).

    This method ensures the domain bounds are within 0 to 360 by handling
    the grid cell that encompasses the prime meridian (e.g., [359, 1]).

    First, it handles the prime meridian cell within the longitude axis bounds
    by splitting the cell into two parts (one east and one west of the prime
    meridian, refer to `_align_lon_bounds_to_360()` for more information). Then
    it concatenates the 360 coordinate point to the longitude coordinates to
    handle the addition of the extra grid cell from the previous step. Finally,
    for each data variable associated with the longitude axis, the value of the
    data variable at the prime meridian cell is concatenated to the data
    variable.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    p_meridian_index : np.ndarray
        An array with a single element representing the index of the prime
        meridian cell.

    Returns
    -------
    xr.Dataset
        The Dataset.
    """
    dim = get_dim_keys(lon_bounds, "X")

    # Create a dataset to store updated longitude variables.
    ds_lon = xr.Dataset()

    # Align the the longitude bounds to the 360 orientation using the prime
    # meridian index. This function splits the grid cell into two parts (east
    # and west), which appends an extra set of bounds for the 360 coordinate.
    ds_lon[lon_bounds.name] = _align_lon_bounds_to_360(lon_bounds, p_meridian_index)

    # After appending the extra set of bounds, update the last coordinate from
    # 0 to 360. Only coordinates on the longitude dimension are updated;
    # auxiliary dimension coordinates (e.g. "bnds") must be left untouched.
    new_coords = {}
    for key, coord in ds_lon.coords.items():
        if dim not in coord.dims:
            continue

        values = coord.values.copy()
        values[-1] = 360
        new_coords[key] = coord.copy(data=values)

    ds_lon = ds_lon.assign_coords(new_coords)

    # Get the data variables related to the longitude axis and concatenate each
    # with the value at the prime meridian.
    for key, var in ds.cf.data_vars.items():
        if key != lon_bounds.name and dim in var.dims:
            # Concatenate the prime meridian cell to the variable
            p_meridian_val = var.isel({dim: p_meridian_index}).copy()
            new_var = xr.concat((var, p_meridian_val), dim=dim)

            # Update the longitude coordinates for the variable.
            new_var[dim] = ds_lon[dim]
            ds_lon[var.name] = new_var

    # Create a new dataset of non-longitude vars and updated longitude vars.
    ds_no_lon = ds.get([v for v in ds.data_vars if dim not in ds[v].dims])  # type: ignore
    ds_final = xr.merge((ds_no_lon, ds_lon))

    # The merge only carries coordinates referenced by its variables, which
    # can drop auxiliary dimension coordinates (e.g. the "bnds" coordinate of
    # longitude bounds). Re-attach every non-longitude coordinate unchanged.
    ds_final = _restore_non_axis_coords(ds_final, ds.coords, dim)

    return ds_final


def _restore_non_axis_coords(
    ds: xr.Dataset, original_coords: xr.Coordinates, axis_dim: str
) -> xr.Dataset:
    """Re-attaches coordinates that do not belong to ``axis_dim``.

    Rebuilding a dataset with ``xr.merge`` can drop auxiliary dimension
    coordinates (e.g. the "bnds" coordinate of longitude bounds) that are not
    referenced as dimension coordinates by the merged variables. This restores
    any such coordinate from the pre-merge snapshot.

    Parameters
    ----------
    ds : xr.Dataset
        The rebuilt dataset potentially missing auxiliary coordinates.
    original_coords : xr.Coordinates
        The coordinate snapshot from before the rebuild.
    axis_dim : str
        The dimension being rebuilt; its own coordinates are left unchanged.

    Returns
    -------
    xr.Dataset
        The dataset with auxiliary coordinates restored.
    """
    for coord_name, coord in original_coords.items():
        if axis_dim in coord.dims:
            continue

        if coord_name not in ds.coords or not ds.coords[coord_name].identical(
            coord
        ):
            ds = ds.assign_coords({coord_name: coord.copy()})

    return ds


def _align_lon_bounds_to_360(
    bounds: xr.DataArray, p_meridian_index: np.ndarray
) -> xr.DataArray:
    """Handles a prime meridian cell to align longitude bounds axis to (0, 360).

    This method ensures the domain bounds are within 0 to 360 by handling
    the grid cell that encompasses the prime meridian (e.g., [359, 1]). In
    this case, calculating longitudinal weights is complicated because the
    weights are determined by the difference of the bounds.

    If this situation exists, the method will split this grid cell into
    two parts (one east and west of the prime meridian). The original
    grid cell will have domain bounds extending east of the prime meridian
    and an extra set of bounds will be concatenated to ``bounds``
    corresponding to the domain bounds west of the prime meridian. For
    instance, a grid cell spanning -1 to 1, will be broken into a cell
    from 0 to 1 and 359 to 360 (or -1 to 0).

    Parameters
    ----------
    bounds : xr.DataArray
        The longitude domain bounds with prime meridian cell.
    p_meridian_index : np.ndarray
        The index of the prime meridian cell.

    Returns
    -------
    xr.DataArray
        The longitude domain bounds with split prime meridian cell.

    Raises
    ------
    ValueError
        If longitude bounds are inclusively between 0 and 360.
    """
    _if_multidim_dask_array_then_load(bounds)

    # Example array: [[359, 1], [1, 90], [90, 180], [180, 359]]
    # Reorient bound to span across zero (i.e., [359, 1] -> [-1, 1]).
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359]]
    bounds[p_meridian_index, 0] = bounds[p_meridian_index, 0] - 360.0

    # Extend the array to nlon+1 by concatenating the grid cell that
    # spans the prime meridian to the end.
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359], [-1, 1]]
    dim = get_dim_keys(bounds, "X")
    bounds = xr.concat((bounds, bounds[p_meridian_index, :]), dim=dim)

    # Add an equivalent bound that spans 360
    # (i.e., [-1, 1] -> [359, 361]) to the end of the array.
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359], [359, 361]]
    repeat_bound = bounds[p_meridian_index, :][0] + 360.0
    bounds[-1, :] = repeat_bound

    # Update the lower-most min and upper-most max bounds to [0, 360].
    # Result: [[0, 1], [1, 90], [90, 180], [180, 359], [359, 360]]
    bounds[p_meridian_index, 0], bounds[-1, 1] = (0.0, 360.0)

    return bounds
