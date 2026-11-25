"""Vertical profile restoration, merging, and target-depth interpolation.

This module supports combining oceanographic profiles that were recorded with
different depth conventions (for example, one profile stored as metres with
depth increasing downward and another as centimetres with depth increasing
upward). Each profile is first restored to a common physical depth axis
(depth below the surface in metres, positive downward) before any merging or
interpolation takes place.

The design follows three rules:

1. Depth restoration happens per profile, and the measurement values and their
   validity masks are reordered together so they never become misaligned.
2. If two valid observations land on the same physical depth and disagree,
   a :class:`ProfileMergeError` identifies the offending profile(s) instead of
   silently keeping one of the layers.
3. Interpolation never extrapolates. Target depths beyond the range spanned by
   valid observations, or separated from a target by an internal gap of
   missing observations, are left missing and the reason is recorded.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Reason labels recorded for target depths that receive no value.
REASON_OUT_OF_RANGE = "out_of_valid_range"
REASON_INTERNAL_GAP = "internal_gap"
REASON_NO_VALID_DATA = "no_valid_observations"

# Factors converting a supported length unit into metres.
_UNIT_TO_METRES: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "dm": 1e-1,
    "decimeter": 1e-1,
    "decimeters": 1e-1,
    "decimetre": 1e-1,
    "decimetres": 1e-1,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
    "centimetre": 1e-2,
    "centimetres": 1e-2,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "millimetre": 1e-3,
    "millimetres": 1e-3,
}

_DIRECTION_SIGN = {
    "down": 1.0,
    "downward": 1.0,
    "positive": 1.0,
    "up": -1.0,
    "upward": -1.0,
    "negative": -1.0,
}


class Profile:
    """A single vertical profile.

    Parameters
    ----------
    values : array-like of float
        Measured values. NaN entries are treated as missing.
    depths : array-like of float
        Recorded level positions in ``units``. The physical meaning of the
        sign is described by ``positive``.
    mask : array-like of bool, optional
        Validity mask, ``True`` for a valid observation. If omitted, every
        finite value is considered valid and NaN values are masked.
    units : str
        Length unit of ``depths`` (e.g. ``"m"`` or ``"cm"``).
    positive : str
        Direction of increasing recorded depth: ``"down"`` (or
        ``"downward"``) / ``"up"`` (or ``"upward"``).
    name : str
        Identifier used in error messages so that contradictory profiles can
        be located.
    """

    def __init__(
        self,
        values: npt.ArrayLike,
        depths: npt.ArrayLike,
        mask: npt.ArrayLike | None = None,
        units: str = "m",
        positive: str = "down",
        name: str = "profile",
    ) -> None:
        self.values = values
        self.depths = depths
        self.mask = mask
        self.units = units
        self.positive = positive
        self.name = name


@dataclass
class RestoredProfile:
    """A profile expressed on physical depth in metres (positive downward).

    Parameters
    ----------
    name : str
        Source profile identifier.
    depths : numpy.ndarray
        Physical depths in metres, positive downward, sorted ascending.
    values : numpy.ndarray
        Measurement values reordered together with ``depths``.
    mask : numpy.ndarray of bool
        Validity mask reordered together with ``depths``.
    """

    name: str
    depths: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    mask: npt.NDArray[np.bool_]


@dataclass
class InterpolationResult:
    """Outcome of interpolating a merged profile onto target depths.

    Parameters
    ----------
    target_depths : numpy.ndarray
        Requested physical depths in metres (positive downward).
    values : numpy.ndarray
        Interpolated values with NaN where no value could be produced.
    mask : numpy.ndarray of bool
        ``True`` exactly where ``values`` contains a produced value.
    reasons : numpy.ndarray of object
        ``None`` for produced values, otherwise one of the ``REASON_*``
        constants explaining why the target depth was left missing.
    merged : RestoredProfile
        The merged profile that was interpolated.
    """

    target_depths: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    mask: npt.NDArray[np.bool_]
    reasons: npt.NDArray[np.object_]
    merged: RestoredProfile


class ProfileMergeError(ValueError):
    """Raised when valid observations on the same physical depth disagree."""


def restore_profile(profile: Profile) -> RestoredProfile:
    """Restore a profile to physical depth in metres and sort by depth.

    Recorded levels are converted to metres using ``profile.units`` and their
    sign is flipped when ``profile.positive`` is upward. Values, depths, and
    the validity mask are then reordered together so they stay aligned.

    Parameters
    ----------
    profile : Profile
        The source profile.

    Returns
    -------
    RestoredProfile
        The profile on ascending physical depth (metres, positive downward).

    Raises
    ------
    ValueError
        If units or direction are unknown, if values and depths have
        different lengths, or if a recorded depth is not finite.
    """
    unit_key = profile.units.strip().lower()
    if unit_key not in _UNIT_TO_METRES:
        raise ValueError(
            f"Unknown depth unit {profile.units!r} in profile "
            f"{profile.name!r}; supported units: {', '.join(sorted(_UNIT_TO_METRES))}"
        )

    direction_key = profile.positive.strip().lower()
    if direction_key not in _DIRECTION_SIGN:
        raise ValueError(
            f"Unknown depth direction {profile.positive!r} in profile "
            f"{profile.name!r}; expected 'down'/'downward' or 'up'/'upward'"
        )

    depths = np.asarray(profile.depths, dtype=np.float64)
    values = np.asarray(profile.values, dtype=np.float64)

    if depths.shape != values.shape or depths.ndim != 1:
        raise ValueError(
            f"Profile {profile.name!r}: 'depths' and 'values' must be 1-D "
            f"arrays of equal length, got depths {depths.shape} and values {values.shape}"
        )
    if not np.all(np.isfinite(depths)):
        raise ValueError(f"Profile {profile.name!r}: all recorded depths must be finite")

    if profile.mask is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(profile.mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(
                f"Profile {profile.name!r}: 'mask' length {mask.shape[0]} does not "
                f"match 'values' length {values.shape[0]}"
            )
        # A NaN measurement cannot be a valid observation.
        mask = mask & np.isfinite(values)

    physical_depths = _DIRECTION_SIGN[direction_key] * depths * _UNIT_TO_METRES[unit_key]

    order = np.argsort(physical_depths, kind="stable")
    return RestoredProfile(
        name=profile.name,
        depths=physical_depths[order],
        values=values[order],
        mask=mask[order],
    )


def merge_profiles(
    profiles: Sequence[Profile],
    depth_tolerance: float = 1e-9,
    value_rtol: float = 1e-9,
    value_atol: float = 1e-9,
) -> RestoredProfile:
    """Restore and merge profiles from the same sea area onto one depth axis.

    Each profile is restored independently, then their samples are combined on
    physical depth (metres, positive downward). Depths within
    ``depth_tolerance`` metres are treated as the same level. When more than
    one valid observation lands on that level and the values are not equal
    (within ``value_rtol``/``value_atol``), a :class:`ProfileMergeError`
    identifies the conflicting profiles rather than silently choosing one.

    Parameters
    ----------
    profiles : sequence of Profile
        Profiles to combine. At least one profile is required.
    depth_tolerance : float
        Absolute tolerance (metres) for considering two depths identical.
    value_rtol, value_atol : float
        Relative and absolute tolerances used to decide whether two valid
        values on the same depth agree.

    Returns
    -------
    RestoredProfile
        The merged profile, named ``"merged"``, sorted by physical depth.

    Raises
    ------
    ValueError
        If no profiles are supplied.
    ProfileMergeError
        If valid observations on the same physical depth contradict each
        other, either within one profile or across profiles.
    """
    profiles = list(profiles)
    if len(profiles) == 0:
        raise ValueError("At least one profile is required to merge")

    restored = [restore_profile(p) for p in profiles]

    all_depths = np.concatenate([r.depths for r in restored])
    all_values = np.concatenate([r.values for r in restored])
    all_mask = np.concatenate([r.mask for r in restored])
    all_names = np.concatenate(
        [np.full(r.depths.shape, r.name, dtype=object) for r in restored]
    )

    order = np.argsort(all_depths, kind="stable")
    all_depths = all_depths[order]
    all_values = all_values[order]
    all_mask = all_mask[order]
    all_names = all_names[order]

    merged_depths: list[float] = []
    merged_values: list[float] = []
    merged_mask: list[bool] = []

    group_start = 0
    n_samples = all_depths.size
    while group_start < n_samples:
        group_end = group_start + 1
        while (
            group_end < n_samples
            and abs(all_depths[group_end] - all_depths[group_start]) <= depth_tolerance
        ):
            group_end += 1

        group_depths = all_depths[group_start:group_end]
        group_values = all_values[group_start:group_end]
        group_mask = all_mask[group_start:group_end]
        group_names = all_names[group_start:group_end]
        valid_idx = np.flatnonzero(group_mask)

        representative_depth = float(group_depths[0])

        if valid_idx.size == 0:
            merged_depths.append(representative_depth)
            merged_values.append(float(group_values[0]))
            merged_mask.append(False)
        else:
            first = valid_idx[0]
            reference_value = group_values[first]
            for idx in valid_idx[1:]:
                if not np.isclose(
                    group_values[idx],
                    reference_value,
                    rtol=value_rtol,
                    atol=value_atol,
                ):
                    details = ", ".join(
                        f"{group_names[i]}={group_values[i]:g}@depth={group_depths[i]:g}m"
                        for i in valid_idx
                    )
                    raise ProfileMergeError(
                        "Contradictory valid observations on physical depth "
                        f"~{representative_depth:g} m from profile(s) "
                        f"{sorted(set(group_names[valid_idx].tolist()))}: {details}"
                    )

            merged_depths.append(representative_depth)
            merged_values.append(float(reference_value))
            merged_mask.append(True)

        group_start = group_end

    return RestoredProfile(
        name="merged",
        depths=np.asarray(merged_depths, dtype=np.float64),
        values=np.asarray(merged_values, dtype=np.float64),
        mask=np.asarray(merged_mask, dtype=np.bool_),
    )


def interpolate_profile(
    profile: RestoredProfile,
    target_depths: npt.ArrayLike,
    depth_tolerance: float = 1e-9,
) -> InterpolationResult:
    """Linearly interpolate a restored profile onto target depths.

    Interpolation is strictly bounded by the valid observations:

    * Target depths outside the range spanned by valid observations are not
      extrapolated and are marked :data:`REASON_OUT_OF_RANGE`.
    * Target depths whose enclosing valid observations are separated by one or
      more missing levels are marked :data:`REASON_INTERNAL_GAP`.
    * If the profile contains no valid observations, every target is marked
      :data:`REASON_NO_VALID_DATA`.

    A target depth coinciding with a valid sample (within
    ``depth_tolerance``) takes that sample's value directly.

    Parameters
    ----------
    profile : RestoredProfile
        A restored (or merged) profile on ascending physical depth.
    target_depths : array-like of float
        Desired physical depths in metres (positive downward).
    depth_tolerance : float
        Absolute tolerance (metres) for matching a target depth to a sample.

    Returns
    -------
    InterpolationResult
        Interpolated values, validity mask, per-target missing reasons, and
        the input profile.

    Raises
    ------
    ValueError
        If a target depth is not finite.
    """
    targets = np.asarray(target_depths, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError(
            f"'target_depths' must be 1-D, got a {targets.ndim}-D array"
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("All target depths must be finite")

    result_values = np.full(targets.shape, np.nan, dtype=np.float64)
    result_mask = np.zeros(targets.shape, dtype=np.bool_)
    reasons = np.full(targets.shape, None, dtype=object)

    valid_positions = np.flatnonzero(profile.mask)
    if valid_positions.size == 0:
        reasons[:] = REASON_NO_VALID_DATA
        return InterpolationResult(
            target_depths=targets,
            values=result_values,
            mask=result_mask,
            reasons=reasons,
            merged=profile,
        )

    valid_depths = profile.depths[valid_positions]
    valid_values = profile.values[valid_positions]
    z_min = valid_depths[0]
    z_max = valid_depths[-1]

    # Mark gaps between consecutive valid samples that contain at least one
    # missing level; interpolation must not cross such a fault.
    gap_below = np.full(valid_positions.shape, False)
    for k in range(1, valid_positions.size):
        if valid_positions[k] > valid_positions[k - 1] + 1:
            gap_below[k] = True

    for i, target in enumerate(targets):
        if target < z_min - depth_tolerance or target > z_max + depth_tolerance:
            reasons[i] = REASON_OUT_OF_RANGE
            continue

        # Exact match to a valid sample.
        exact = np.flatnonzero(np.abs(valid_depths - target) <= depth_tolerance)
        if exact.size > 0:
            result_values[i] = valid_values[exact[0]]
            result_mask[i] = True
            continue

        if target < z_min or target > z_max:
            # Within tolerance of an edge but not matching it: do not
            # extrapolate.
            reasons[i] = REASON_OUT_OF_RANGE
            continue

        right = int(np.searchsorted(valid_depths, target, side="left"))
        left = right - 1

        if gap_below[right]:
            reasons[i] = REASON_INTERNAL_GAP
            continue

        z0, z1 = valid_depths[left], valid_depths[right]
        v0, v1 = valid_values[left], valid_values[right]
        weight = (target - z0) / (z1 - z0)
        result_values[i] = v0 + weight * (v1 - v0)
        result_mask[i] = True

    return InterpolationResult(
        target_depths=targets,
        values=result_values,
        mask=result_mask,
        reasons=reasons,
        merged=profile,
    )


def merge_and_interpolate(
    profiles: Sequence[Profile],
    target_depths: npt.ArrayLike,
    depth_tolerance: float = 1e-9,
    value_rtol: float = 1e-9,
    value_atol: float = 1e-9,
) -> InterpolationResult:
    """Restore profiles by source, merge them, then interpolate target depths.

    This is the convenience entry point combining :func:`merge_profiles` and
    :func:`interpolate_profile`. Restoration to physical depth always happens
    per profile before merging, so mixed units (metres/centimetres) and mixed
    directions (downward/upward positive) are handled correctly.

    Parameters
    ----------
    profiles : sequence of Profile
        Profiles from the same sea area, potentially in different units and
        depth conventions.
    target_depths : array-like of float
        Desired physical depths in metres (positive downward).
    depth_tolerance : float
        Absolute tolerance (metres) for treating depths as identical.
    value_rtol, value_atol : float
        Tolerances for detecting contradictory values on the same depth.

    Returns
    -------
    InterpolationResult
        Interpolated values with a mask and per-target missing reasons.

    Raises
    ------
    ProfileMergeError
        If valid observations on the same physical depth contradict each
        other.
    """
    merged = merge_profiles(
        profiles,
        depth_tolerance=depth_tolerance,
        value_rtol=value_rtol,
        value_atol=value_atol,
    )
    return interpolate_profile(merged, target_depths, depth_tolerance=depth_tolerance)
