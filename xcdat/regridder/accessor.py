from collections.abc import Mapping
from typing import Any, Literal

import xarray as xr

from xcdat.axis import CFAxisKey, get_axis_coord, get_coords_by_name, get_dim_coords
from xcdat.bounds import create_bounds
from xcdat.regridder import regrid2, xesmf, xgcm
from xcdat.regridder.base import BaseRegridder
from xcdat.regridder.grid import _validate_grid_has_single_axis_dim

#: Type alias for explicit per-call axis-to-coordinate selections.
CoordKeys = Mapping[CFAxisKey, str]

HorizontalRegridTools = Literal["xesmf", "regrid2"]
HORIZONTAL_REGRID_TOOLS: dict[str, type[BaseRegridder]] = {
    "regrid2": regrid2.Regrid2Regridder,
    "xesmf": xesmf.XESMFRegridder,
}

VerticalRegridTools = Literal["xgcm"]
VERTICAL_REGRID_TOOLS: dict[str, type[BaseRegridder]] = {"xgcm": xgcm.XGCMRegridder}


@xr.register_dataset_accessor(name="regridder")
class RegridderAccessor:
    """
    An accessor class that provides regridding attributes and methods for
    xarray Datasets through the ``.regridder`` attribute.

    Examples
    --------

    Import xCDAT:

    >>> import xcdat

    Use RegridderAccessor class:

    >>> ds = xcdat.open_dataset("...")
    >>>
    >>> ds.regridder.<attribute>
    >>> ds.regridder.<method>
    >>> ds.regridder.<property>

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset to attach this accessor.
    """

    def __init__(self, dataset: xr.Dataset):
        self._ds: xr.Dataset = dataset

    @property
    def grid(self) -> xr.Dataset:
        """
        Extract the `X`, `Y`, and `Z` axes from the Dataset and return a new
        ``xr.Dataset``.

        Returns
        -------
        xr.Dataset
            Containing grid axes.

        Raises
        ------
        ValueError
            If axis dimension coordinate variable is not correctly identified.
        ValueError
            If axis has multiple dimensions (only one is expected).

        Examples
        --------

        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Extract grid from dataset:

        >>> grid = ds.regridder.grid
        """
        return _obj_to_grid_ds(self._ds)

    def horizontal(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        tool: HorizontalRegridTools = "xesmf",
        coord_keys: CoordKeys | None = None,
        **options: Any,
    ) -> xr.Dataset:
        """
        Transform ``data_var`` to ``output_grid``.

        When might ``Regrid2`` be preferred over ``xESMF``?

        If performing conservative regridding from a high/medium resolution lat/lon grid to a
        coarse lat/lon target, ``Regrid2`` may provide better results as it assumes grid cells
        with constant latitudes and longitudes while ``xESMF`` assumes the cells are connected
        by Great Circles [1]_.

        Supported tools, methods and grids:

        - xESMF (https://xesmf.readthedocs.io/en/latest/)
           - Methods: Bilinear, Conservative, Conservative Normed, Patch, Nearest s2d, or Nearest d2s.
           - Grids: Rectilinear, or Curvilinear.
           - Find options at :py:func:`xcdat.regridder.xesmf.XESMFRegridder`
        - Regrid2
           - Methods: Conservative
           - Grids: Rectilinear
           - Find options at :py:func:`xcdat.regridder.regrid2.Regrid2Regridder`

        Parameters
        ----------
        data_var: str
            Name of the variable to transform.
        output_grid : xr.Dataset
            Grid to transform ``data_var`` to.
        tool : str
            Name of the tool to use.
        coord_keys : CoordKeys | None, optional
            Explicit axis-to-coordinate mapping for disambiguating merged
            multi-grid datasets, e.g. ``{"X": "olon", "Y": "olat"}``. The
            source grid (coordinates, bounds, and mask) is then built from
            those coordinates only. Selection applies to this call only and
            does not affect how any other data variable is resolved. By
            default None.
        **options : Any
            These options are passed directly to the ``tool``. See specific
            regridder for available options.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` transformed to the ``output_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        References
        ----------
        .. [1] https://earthsystemmodeling.org/docs/release/ESMF_8_1_0/ESMF_refdoc/node5.html#SECTION05012900000000000000

        Examples
        --------

        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Create output grid:

        >>> output_grid = xcdat.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        Regrid variable using "xesmf":

        >>> output_data = ds.regridder.horizontal("ts", output_grid, tool="xesmf", method="bilinear")

        Regrid variable using "regrid2":

        >>> output_data = ds.regridder.horizontal("ts", output_grid, tool="regrid2")
        """
        try:
            regrid_tool = HORIZONTAL_REGRID_TOOLS[tool]
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices {list(HORIZONTAL_REGRID_TOOLS)}"
            ) from e

        input_grid = _get_input_grid(
            self._ds,
            data_var,
            ["X", "Y"],
            multidim=regrid_tool.can_handle_multidim(),
            coord_keys=coord_keys,
        )
        regridder = regrid_tool(input_grid, output_grid, **options)
        output_ds = regridder.horizontal(data_var, self._ds)

        return output_ds

    def vertical(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        tool: VerticalRegridTools = "xgcm",
        coord_keys: CoordKeys | None = None,
        **options: Any,
    ) -> xr.Dataset:
        """
        Transform ``data_var`` to ``output_grid``.

        Supported tools:

        - xgcm (https://xgcm.readthedocs.io/en/latest/index.html)
           - Methods: Linear, Conservative, Log
           - Find options at :py:func:`xcdat.regridder.xgcm.XGCMRegridder`

        Parameters
        ----------
        data_var: str
            Name of the variable to transform.
        output_grid : xr.Dataset
            Grid to transform ``data_var`` to.
        tool : str
            Name of the tool to use.
        coord_keys : CoordKeys | None, optional
            Explicit axis-to-coordinate mapping for disambiguating merged
            multi-grid datasets. Selection applies to this call only. By
            default None.
        **options : Any
            These options are passed directly to the ``tool``. See specific
            regridder for available options.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` transformed to the ``output_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        Examples
        --------

        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Create output grid:

        >>> output_grid = xcdat.create_grid(lev=np.linspace(1000, 1, 20))

        Regrid variable using "xgcm":

        >>> output_data = ds.regridder.vertical("so", output_grid, method="linear")
        """
        try:
            regrid_tool = VERTICAL_REGRID_TOOLS[tool]
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices "
                f"{list(VERTICAL_REGRID_TOOLS)}"
            ) from e

        input_grid = _get_input_grid(
            self._ds, data_var, ["Z"], coord_keys=coord_keys
        )
        regridder = regrid_tool(input_grid, output_grid, **options)
        output_ds = regridder.vertical(data_var, self._ds)

        return output_ds


def _obj_to_grid_ds(
    obj: xr.Dataset | xr.DataArray,
    multidim: bool = False,
    coord_keys: Mapping[CFAxisKey, str] | None = None,
    target_var: str | None = None,
) -> xr.Dataset:
    """
    Convert an xarray object to a new Dataset containing axis coordinates and
    bounds.

    This function extracts axis coordinates and bounds for the specified
    axes ("X", "Y", "Z") from the input object and creates a new xarray
    Dataset. If bounds are missing for an axis, they are added to the
    output Dataset.

    When ``obj`` is a ``xr.DataArray`` or ``coord_keys`` is supplied, the
    variable-aware resolver (:func:`xcdat.axis.get_axis_coord`) is used so
    that merged multi-grid datasets never borrow another variable's
    coordinates or bounds. ``target_var`` pins resolution to a specific
    data variable while other axes (e.g., "Z") on the source Dataset are
    still retained.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        The input xarray object containing the data and attributes.
    multidim : bool, optional
        Whether multidimensional coordinates are allowed.
    coord_keys : Mapping[CFAxisKey, str] | None, optional
        Explicit axis-to-coordinate mapping for this call.
    target_var : str | None, optional
        The data variable whose grid should be resolved when ``obj`` is a
        Dataset with multiple variables.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset containing the axis coordinates, bounds, and
        attributes from the input object.

    Notes
    -----
    - The function ensures that bounds are only added for axes that do not
      already have them. This avoids duplicating bounds for axes with
      multiple coordinates (e.g., curvilinear grids).
    - The `xr.set_options(keep_attrs=True)` context is used to preserve
      attributes from the input object in the output Dataset.
    """
    axis_names: list[CFAxisKey] = ["X", "Y", "Z"]

    axis_coords: dict[str, xr.DataArray] = {}
    axis_bounds: dict[str, xr.DataArray] = {}
    axis_has_bounds: dict[CFAxisKey, bool] = {}

    with xr.set_options(keep_attrs=True):
        for axis in axis_names:
            if isinstance(obj, xr.DataArray) or coord_keys is not None:
                coord, bounds = _get_resolved_axis_coord_and_bounds(
                    obj,
                    axis,
                    multidim=multidim,
                    coord_key=(coord_keys or {}).get(axis),
                    target_var=target_var,
                )
                # Axes absent from the target variable (e.g., a "Z" axis on
                # a 2-D variable) are still part of the source grid and were
                # historically extracted at the dataset level.
                if coord is None and isinstance(obj, xr.Dataset):
                    coord, bounds = _get_axis_coord_and_bounds(
                        obj, axis, multidim=multidim
                    )
            else:
                coord, bounds = _get_axis_coord_and_bounds(
                    obj, axis, multidim=multidim
                )

            if coord is not None:
                axis_coords[str(coord.name)] = coord

                if bounds is not None:
                    axis_bounds[str(bounds.name)] = bounds
                    axis_has_bounds[axis] = True
                else:
                    axis_has_bounds[axis] = False

    # Create a new dataset with coordinates and bounds
    output_ds = xr.Dataset(
        coords=axis_coords,
        data_vars=axis_bounds,
        attrs=obj.attrs,
    )

    # Multidimensional coordinates bounds generation is not supported
    if multidim:
        return output_ds

    # Add bounds only for axes that do not already have them. This
    # prevents multiple sets of bounds being added for the same axis.
    # For example, curvilinear grids can have multiple coordinates for the
    # same axis (e.g., (nlat, lat) for X and (nlon, lon) for Y). We only
    # need lat_bnds and lon_bnds for the X and Y axes, respectively, and not
    # nlat_bnds and nlon_bnds.

    for axis, has_bounds in axis_has_bounds.items():
        if not has_bounds:
            output_ds = output_ds.bounds.add_bounds(axis=axis)

    return output_ds


def _get_resolved_axis_coord_and_bounds(
    obj: xr.Dataset | xr.DataArray,
    axis: CFAxisKey,
    multidim: bool = False,
    coord_key: str | None = None,
    target_var: str | None = None,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """Resolve one axis coordinate (and its bounds) on a specific variable.

    Coordinates are resolved on the target data variable, while bounds are
    looked up on the parent Dataset because bounds are stored as dataset-level
    data variables rather than DataArray coordinates.
    """
    if isinstance(obj, xr.DataArray):
        da = obj
        bounds_source: xr.Dataset | xr.DataArray = obj
    else:
        da = obj[target_var] if target_var is not None else _single_var(obj)
        bounds_source = obj

    try:
        coord_var = get_axis_coord(
            da, axis, coord_key=coord_key, multidim=multidim
        )
    except KeyError:
        return None, None
    if not multidim:
        _validate_grid_has_single_axis_dim(axis, coord_var)

    if coord_var.size == 1:
        raise ValueError(
            f"Coordinate '{coord_var.name}' is a singleton and cannot be used."
        )

    bounds_var = None
    bounds_key = coord_var.attrs.get("bounds")
    if bounds_key:
        if isinstance(bounds_source, xr.Dataset):
            bounds_var = bounds_source.data_vars.get(bounds_key)
            if bounds_var is None:
                bounds_var = bounds_source.coords.get(bounds_key)
        else:
            bounds_var = bounds_source.coords.get(bounds_key)
        # Only 1-D coordinates support bounds generation; 2-D curvilinear
        # coordinates must provide their own bounds.
        if bounds_var is None and coord_var.ndim == 1:
            bounds_var = create_bounds(axis, coord_var)

    return coord_var, bounds_var


def _single_var(ds: xr.Dataset) -> xr.DataArray:
    """Return the single non-bounds data variable of a small grid dataset."""
    non_bounds = [name for name in ds.data_vars if name not in ds.bounds.keys]

    if len(non_bounds) != 1:
        raise ValueError(
            f"Expected a single data variable to resolve grid coordinates, "
            f"found {non_bounds}."
        )

    return ds[non_bounds[0]]


def _get_axis_coord_and_bounds(
    obj: xr.Dataset | xr.DataArray, axis: CFAxisKey, multidim: bool = False
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    try:
        coord_var = get_coords_by_name(obj, axis)
        if coord_var.size == 1:
            raise ValueError(
                f"Coordinate '{coord_var}' is a singleton and cannot be used."
            )
    except (ValueError, KeyError):
        try:
            coord_var = get_dim_coords(obj, axis, multidim=multidim)  # type: ignore
            _validate_grid_has_single_axis_dim(axis, coord_var)
        except KeyError:
            coord_var = None

    if coord_var is None:
        return None, None

    bounds_var = None
    bounds_key = coord_var.attrs.get("bounds")
    if bounds_key:
        try:
            bounds_var = obj.get(bounds_key)
        except AttributeError:
            bounds_var = create_bounds(axis, coord_var)

    return coord_var, bounds_var


def _get_input_grid(
    ds: xr.Dataset,
    data_var: str,
    dup_check_dims: list[CFAxisKey],
    multidim: bool = False,
    coord_keys: Mapping[CFAxisKey, str] | None = None,
):
    """
    Extract the grid for ``data_var`` from ``ds``.

    The grid is built from the coordinates resolved on the data variable
    itself (via :func:`xcdat.axis.get_axis_coord`), along with each
    coordinate's own bounds. Coordinate variables that map to the same axis
    but belong to a different grid (e.g., "lon" vs "olon" on same-length
    dimensions) are dropped instead of being silently mixed in. Any "mask"
    variable is only kept when its dimensions match the resolved grid.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to extract grid from.
    data_var : str
        Name of target data variable.
    dup_check_dims : list[CFAxisKey]
        List of axes to resolve.
    multidim : bool, optional
        Whether multidimensional coordinates are allowed.
    coord_keys : Mapping[CFAxisKey, str] | None, optional
        Explicit axis-to-coordinate mapping for this call only.

    Returns
    -------
    xr.Dataset
        Dataset containing grid dataset.
    """
    da = ds[data_var]

    resolved: dict[CFAxisKey, xr.DataArray] = {}
    for axis in dup_check_dims:
        resolved[axis] = get_axis_coord(
            da, axis, coord_key=(coord_keys or {}).get(axis), multidim=multidim
        )
    resolved_names = {str(coord.name) for coord in resolved.values()}
    resolved_keymap = {axis: str(coord.name) for axis, coord in resolved.items()}
    resolved_dims: set[str] = set()
    for coord in resolved.values():
        resolved_dims.update(coord.dims)
    resolved_ranks = {coord.ndim for coord in resolved.values()}

    # Other coordinate variables mapped to the same axes belong to a
    # different grid (e.g., an auxiliary ocean grid on shared dim names).
    # Index coordinates of a different rank (e.g., 1-D "nlat"/"nlon"
    # indexes backing 2-D curvilinear "lat"/"lon") are part of the same
    # grid and must be retained.
    index_names = set(ds.indexes)
    rival_coord_names: set[str] = set()
    for axis in dup_check_dims:
        for key in _get_all_axis_coord_keys_safe(ds, axis, multidim):
            if key in resolved_names:
                continue
            coord = ds.coords[key]
            if not set(coord.dims) & set(da.dims):
                continue
            if (
                key in index_names
                and set(coord.dims) <= resolved_dims
                and coord.ndim not in resolved_ranks
            ):
                continue
            rival_coord_names.add(key)

    # Drop rival coordinate variables (and only them) so the source grid
    # describes the selected variable exclusively.
    to_drop = [name for name in rival_coord_names if name in ds.coords]
    stripped = ds.drop_vars(to_drop) if to_drop else ds

    # Build the grid from the full stripped dataset, pinning resolution to
    # the target variable so unrelated axes (e.g., "Z") are still retained.
    grid = _obj_to_grid_ds(
        stripped,
        multidim=multidim,
        coord_keys=resolved_keymap,
        target_var=data_var,
    )

    # Preserve a mask only when it is defined on the resolved grid dims.
    grid_dims = set(grid.dims)
    matching_masks = [
        name
        for name in ds.data_vars
        if name == "mask" and set(ds[name].dims) <= grid_dims
        and set(ds[name].dims) & grid_dims
    ]

    if len(matching_masks) > 1:
        raise ValueError(
            f"More than one 'mask' variable matches the resolved grid "
            f"dimensions {sorted(grid_dims)}: {matching_masks}. Rename the "
            "unrelated masks or keep the mask for the selected variable."
        )
    if matching_masks:
        grid["mask"] = ds[matching_masks[0]].copy()

    return grid


def _get_all_axis_coord_keys_safe(
    ds: xr.Dataset, axis: CFAxisKey, multidim: bool
) -> list[str]:
    """List axis coordinate keys in the dataset, tolerating ambiguity."""
    from xcdat.axis import _get_all_coord_keys

    keys = set(_get_all_coord_keys(ds, axis))

    if multidim:
        return sorted(name for name in keys if ds.coords[name].ndim > 0)

    return sorted(
        name for name in keys if name in ds.indexes or ds.coords[name].ndim == 1
    )
