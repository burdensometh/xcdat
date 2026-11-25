"""Unit harmonization for data variables opened from multiple files.

The same-named variable produced by different providers can be encoded in
different units (e.g., surface air temperature in degrees Celsius in one file
and Kelvin in another), with file-specific ``_FillValue``/``missing_value``
sentinels and valid-range attributes. This module performs explicit,
missing-value-aware unit conversions per source dataset so that identically
named variables can be concatenated without one file's attributes masking a
conflict in another file.

Only temperature units are currently supported: Kelvin, degrees Celsius, and
degrees Fahrenheit. Conversions are opt-in – callers must explicitly declare
the target unit for each variable; source units are never guessed.
"""

from collections.abc import Mapping

import xarray as xr

# Mapping of accepted unit spellings (case-insensitive) to canonical labels.
# The canonical labels follow the CF conventions and are written back to the
# converted variable's ``units`` attribute. Both space- and underscore-separated
# spellings are accepted (e.g., "degree celsius" and "degree_celsius").
_TEMPERATURE_UNIT_ALIASES: dict[str, str] = {
    "k": "K",
    "kelvin": "K",
    "degrees k": "K",
    "degree k": "K",
    "degrees_k": "K",
    "degree_k": "K",
    "degrees kelvin": "K",
    "degree kelvin": "K",
    "degrees_kelvin": "K",
    "degree_kelvin": "K",
    "degc": "degC",
    "c": "degC",
    "celsius": "degC",
    "degree c": "degC",
    "degrees c": "degC",
    "degree_c": "degC",
    "degrees_c": "degC",
    "degree celsius": "degC",
    "degrees celsius": "degC",
    "degree_celsius": "degC",
    "degrees_celsius": "degC",
    "degf": "degF",
    "f": "degF",
    "fahrenheit": "degF",
    "degree f": "degF",
    "degrees f": "degF",
    "degree_f": "degF",
    "degrees_f": "degF",
    "degree fahrenheit": "degF",
    "degrees fahrenheit": "degF",
    "degree_fahrenheit": "degF",
    "degrees_fahrenheit": "degF",
}

#: Canonical temperature units supported by this module.
SUPPORTED_TEMPERATURE_UNITS: tuple[str, ...] = ("K", "degC", "degF")

#: Attributes describing file-specific encodings that cease to be meaningful
#: once missing values have been decoded to NaN and the unit is converted.
_ENCODING_SPECIFIC_ATTRS: tuple[str, ...] = (
    "_FillValue",
    "missing_value",
    "valid_range",
    "valid_min",
    "valid_max",
)

#: Attribute recording the unit each source file was originally encoded in.
SOURCE_UNITS_ATTR = "source_units"

#: Attribute recording the unit the variable was converted to.
TARGET_UNITS_ATTR = "target_units"


def _source_file(dataset: xr.Dataset) -> str:
    """Returns the source file path recorded on a dataset, if available.

    Parameters
    ----------
    dataset : xr.Dataset
        A dataset opened by xarray, which records its originating file in
        ``encoding["source"]`` during multi-file reads.

    Returns
    -------
    str
        The source file path, or ``"<unknown>"`` if it is not recorded.
    """
    source = dataset.encoding.get("source")
    return str(source) if source is not None else "<unknown>"


def build_source_units_record(
    dataset: xr.Dataset, harmonize_units: Mapping[str, str]
) -> dict[str, str]:
    """Builds a ``{variable: "file: unit; file: unit"}`` provenance record.

    The record captures the original ``units`` attribute of every converted
    variable in every source file. xarray keeps only the first dataset's
    variable-level attributes when concatenating, so this separate record is
    required to prevent a conflict in a later file from being masked by the
    first file's attributes.

    Parameters
    ----------
    dataset : xr.Dataset
        A single source dataset (its ``encoding["source"]`` identifies the
        file).
    harmonize_units : Mapping[str, str]
        The mapping of data variable key to target unit that was applied.

    Returns
    -------
    dict[str, str]
        A mapping of variable key to a single string recording the source file
        and that file's original unit.
    """
    source = _source_file(dataset)
    record: dict[str, str] = {}

    for var_key in harmonize_units:
        if var_key not in dataset.data_vars:
            continue

        source_units = dataset[var_key].attrs.get("units")
        if source_units is None:
            source_label = "unknown"
        else:
            try:
                source_label = normalize_temperature_unit(source_units)
            except ValueError:
                # Preserve the unrecognized label verbatim; validation and the
                # file-located error are raised during conversion.
                source_label = f"unrecognized ({source_units})"
        record[var_key] = f"{source}: {source_label}"

    return record


def merge_source_units_records(
    records: list[dict[str, str]],
) -> dict[str, str]:
    """Merges per-file provenance records into dataset-level attributes.

    Entries are sorted by source file so the record is identical regardless of
    the order files were passed in.

    Parameters
    ----------
    records : list[dict[str, str]]
        Per-file records produced by :func:`build_source_units_record`.

    Returns
    -------
    dict[str, str]
        A mapping of attribute name (``"<var>_source_units"``) to a
        semicolon-delimited string of every source file and its original unit.
    """
    grouped: dict[str, list[str]] = {}
    for record in records:
        for var_key, entry in record.items():
            grouped.setdefault(var_key, []).append(entry)

    return {
        f"{var_key}_source_units": "; ".join(sorted(set(entries)))
        for var_key, entries in grouped.items()
    }


def normalize_temperature_unit(units: str) -> str:
    """Normalizes a temperature unit string to its canonical CF label.

    Parameters
    ----------
    units : str
        A temperature unit string (e.g., ``"K"``, ``"kelvin"``, ``"degC"``,
        ``"Celsius"``, ``"degF"``). Matching is case-insensitive.

    Returns
    -------
    str
        The canonical label: one of ``"K"``, ``"degC"``, or ``"degF"``.

    Raises
    ------
    ValueError
        If ``units`` is not a recognized temperature unit.
    """
    if not isinstance(units, str):
        raise ValueError(
            f"Expected a string unit label, but received {units!r} "
            f"({type(units).__name__})."
        )

    key = units.strip().lower()
    canonical = _TEMPERATURE_UNIT_ALIASES.get(key)

    if canonical is None:
        raise ValueError(
            f"Unsupported temperature unit '{units}'. Supported units are "
            f"{', '.join(SUPPORTED_TEMPERATURE_UNITS)} (and common aliases "
            "such as 'Kelvin', 'Celsius', or 'Fahrenheit')."
        )

    return canonical


def convert_temperature_dataarray(
    da: xr.DataArray, target_units: str
) -> xr.DataArray:
    """Converts a temperature DataArray to ``target_units``.

    Missing values are expected to already be decoded to NaN by xarray's
    ``mask_and_scale`` decoding. Arithmetic propagates NaN, so sentinel values
    are never transformed as if they were valid measurements. Conversion is
    lazy: Dask-backed arrays stay Dask-backed.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to convert. It must have a ``units`` attribute set to a
        supported temperature unit.
    target_units : str
        The target unit (e.g., ``"K"`` or ``"degC"``).

    Returns
    -------
    xr.DataArray
        A converted DataArray with updated ``units`` and provenance attributes.

    Raises
    ------
    ValueError
        If the source ``units`` attribute is missing or not a supported
        temperature unit, or if ``target_units`` is not supported.
    """
    source_units = da.attrs.get("units")
    if source_units is None:
        raise ValueError(
            f"Variable '{da.name}' has no 'units' attribute, so its unit "
            "cannot be converted."
        )

    source_canonical = normalize_temperature_unit(source_units)
    target_canonical = normalize_temperature_unit(target_units)

    if source_canonical == target_canonical:
        converted = da.copy()
    else:
        converted = _apply_temperature_conversion(
            da, source_canonical, target_canonical
        )

    converted.attrs = dict(da.attrs)
    # File-specific fill values and valid ranges refer to the original encoding
    # and are invalid once values are converted.
    for attr in _ENCODING_SPECIFIC_ATTRS:
        converted.attrs.pop(attr, None)

    converted.attrs[SOURCE_UNITS_ATTR] = source_canonical
    converted.attrs[TARGET_UNITS_ATTR] = target_canonical
    converted.attrs["units"] = target_canonical

    return converted


def harmonize_dataset_units(
    dataset: xr.Dataset, harmonize_units: Mapping[str, str]
) -> xr.Dataset:
    """Harmonizes the units of data variables within a single source dataset.

    This is intended to run on each dataset opened from an individual file
    (e.g., via the ``preprocess`` hook of ``open_mfdataset``), so missing
    values are identified using that file's own fill-value encoding before any
    conversion is applied. Variables listed in ``harmonize_units`` that are not
    present in this dataset are silently skipped – a variable may only exist in
    a subset of the input files.

    Parameters
    ----------
    dataset : xr.Dataset
        A dataset decoded with ``mask_and_scale`` enabled (the default), so
        fill values are represented as NaN.
    harmonize_units : Mapping[str, str]
        A mapping of data variable key to the explicit target temperature unit
        (e.g., ``{"tas": "K"}``).

    Returns
    -------
    xr.Dataset
        A dataset with the listed variables converted to their target units.

    Raises
    ------
    ValueError
        If a listed variable's source ``units`` attribute is missing or
        unrecognized, or if a requested target unit is not supported. The
        offending variable name and, when available, source file
        (``dataset.encoding["source"]``) are included in the message so the
        problematic file can be located.
    """
    ds = dataset
    source = dataset.encoding.get("source")
    source_msg = f" from source '{source}'" if source else ""

    for var_key, target_units in harmonize_units.items():
        if var_key not in ds.data_vars:
            continue

        da = ds[var_key]
        source_units = da.attrs.get("units")
        if source_units is None:
            raise ValueError(
                f"Failed to harmonize units for variable '{var_key}'{source_msg}: "
                "the variable has no 'units' attribute, so its unit cannot be "
                "converted."
            )

        try:
            normalize_temperature_unit(source_units)
        except ValueError as err:
            raise ValueError(
                f"Failed to harmonize units for variable '{var_key}'{source_msg}: "
                f"{err}"
            ) from err

        converted = convert_temperature_dataarray(da, target_units)
        ds = ds.assign({var_key: converted})

    return ds


def validate_harmonize_units(harmonize_units: Mapping[str, str]) -> None:
    """Validates a ``harmonize_units`` mapping up front.

    Target units are checked before any files are opened so that an invalid
    caller request fails fast rather than after partial processing.

    Parameters
    ----------
    harmonize_units : Mapping[str, str]
        A mapping of data variable key to target temperature unit.

    Raises
    ------
    ValueError
        If a variable key is not a string or a target unit is unsupported.
    """
    for var_key, target_units in harmonize_units.items():
        if not isinstance(var_key, str):
            raise ValueError(
                f"harmonize_units keys must be variable name strings, "
                f"but received {var_key!r} ({type(var_key).__name__})."
            )
        # Raises ValueError for unsupported target units.
        normalize_temperature_unit(target_units)


def _apply_temperature_conversion(
    da: xr.DataArray, source_units: str, target_units: str
) -> xr.DataArray:
    """Applies the temperature conversion between two canonical units.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to convert.
    source_units : str
        The canonical source unit.
    target_units : str
        The canonical target unit.

    Returns
    -------
    xr.DataArray
        The converted DataArray.
    """
    if source_units == "K":
        if target_units == "degC":
            return da - 273.15
        # K -> degF
        return (da - 273.15) * 9.0 / 5.0 + 32.0

    if source_units == "degC":
        if target_units == "K":
            return da + 273.15
        # degC -> degF
        return da * 9.0 / 5.0 + 32.0

    # source_units == "degF"
    if target_units == "degC":
        return (da - 32.0) * 5.0 / 9.0
    # degF -> K
    return (da - 32.0) * 5.0 / 9.0 + 273.15
