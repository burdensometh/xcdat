"""One-to-one matching of ground-station observations to grid samples.

This module pairs ground-station observations with gridded product samples so
that biases can be computed from the selected pairs. A station observation and
a grid sample are eligible to pair when they belong to the same product period
and simultaneously satisfy:

* a great-circle (spherical) distance tolerance, and
* an absolute time-difference tolerance.

Within each period the pairing is a one-to-one assignment: no observation and
no sample may be used twice in the same period. The assignment is selected by
lexicographic optimization:

1. maximize the number of valid (feasible) pairs,
2. minimize the total great-circle distance of those pairs,
3. break remaining ties deterministically using stable sample (and then
   observation) identity.

The third level is encoded as a sub-millimeter fractional term per edge, so it
only affects assignments whose total distances are equal at floating-point
resolution; genuinely closer pairings always win regardless of identifiers.

Because the tie-breaker uses ranks derived from the stable identifiers rather
than input ordering or array positions, shuffling the input rows does not
change the result. Great-circle distances are computed on the sphere, which is
also invariant to whether longitudes are expressed across the antimeridian
(e.g. ``179.9`` vs. ``-180.1``).
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from xcdat._logger import _setup_custom_logger

logger = _setup_custom_logger(__name__)

EARTH_RADIUS_KM = 6371.0088

# Rejection reason vocabulary.
REASON_DISTANCE = "outside distance tolerance"
REASON_TIME = "outside time tolerance"
REASON_PERIOD = "no sample in the same product period"
REASON_ASSIGNMENT = "not selected in the optimal assignment"

# Feasible edge costs encode a strict lexicographic ordering
# (feasibility > total distance > sample identity > observation identity)
# using one scalar cost per edge. Infeasible edges receive a penalty above
# every feasible assignment total so maximizing feasible pairs always takes
# precedence over the distance objective.
_INFEASIBLE_FACTOR = 4
# Fractional identity tie-break, in kilometers. It stays below one millimeter
# per edge, so it can only reorder assignments whose total distances are
# equal to sub-millimeter precision; exactly equal distances (the intended
# tie case) are always resolved by stable identity.
_IDENTITY_EPSILON = 1.0e-6
# Guard for the rank-based tie-break: beyond this many records per period the
# smallest rank step approaches floating-point resolution. Real station or
# sample panels are far smaller, and the guard fails loudly instead of
# silently reordering results.
_MAX_PERIOD_RECORDS = 100_000

_MATCHED_COLUMNS = [
    "station_id",
    "sample_id",
    "period",
    "distance_km",
    "time_delta",
    "observation_value",
    "sample_value",
    "bias",
]
_UNMATCHED_STATION_COLUMNS = [
    "station_id",
    "period",
    "reason",
    "time_delta",
]
_UNMATCHED_SAMPLE_COLUMNS = [
    "sample_id",
    "period",
    "reason",
    "time_delta",
]


def great_circle_distance_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
    radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Compute great-circle distances between points on a sphere.

    Distances use the haversine formula, which works for any longitude
    convention: points near the antimeridian (e.g. ``lon = 179.9`` and
    ``lon = -179.9``) are correctly treated as close together.

    Parameters
    ----------
    lat1, lon1 : np.ndarray or float
        Latitude(s) and longitude(s) of the first point(s) in degrees.
    lat2, lon2 : np.ndarray or float
        Latitude(s) and longitude(s) of the second point(s) in degrees.
    radius_km : float, optional
        Sphere radius in kilometers, by default ``EARTH_RADIUS_KM`` (the mean
        Earth radius).

    Returns
    -------
    np.ndarray
        Great-circle distance(s) in kilometers. Broadcast inputs follow NumPy
        broadcasting rules.

    Raises
    ------
    ValueError
        If a latitude is outside ``[-90, 90]``.
    """
    lat1a = np.asarray(lat1, dtype=np.float64)
    lat2a = np.asarray(lat2, dtype=np.float64)
    if np.any(np.abs(lat1a) > 90.0) or np.any(np.abs(lat2a) > 90.0):
        raise ValueError("Latitudes must be within [-90, 90] degrees.")

    lat1r, lon1r = np.radians(lat1a), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2a), np.radians(lon2)

    dlat = lat2r - lat1r
    # Wrapping the longitude difference to [-pi, pi) makes the calculation
    # antimeridian-agnostic.
    dlon = np.mod((lon2r - lon1r) + np.pi, 2.0 * np.pi) - np.pi

    hav = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    # Clamp guards against tiny floating-point overshoots outside [0, 1].
    angle = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    return radius_km * angle


def match_stations_to_samples(
    observations: pd.DataFrame,
    samples: pd.DataFrame,
    distance_tolerance_km: float,
    time_tolerance: pd.Timedelta | str,
    period_freq: str | None = None,
    *,
    period_column: str | None = None,
    station_id_col: str = "station_id",
    sample_id_col: str = "sample_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
    time_col: str = "time",
    value_col: str = "value",
    sample_lat_col: str | None = None,
    sample_lon_col: str | None = None,
    sample_time_col: str | None = None,
    sample_value_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match ground-station observations one-to-one to grid samples.

    Matching is performed independently within each product period. A pair is
    feasible only when both the great-circle distance and the absolute time
    difference are within their tolerances. The optimal assignment first
    maximizes the number of feasible pairs, then minimizes total
    great-circle distance, and finally breaks ties by stable sample identity
    (then observation identity). Results are therefore independent of input
    row order and of longitude conventions across the antimeridian.

    Parameters
    ----------
    observations : pd.DataFrame
        Station observations. Must contain identifier, latitude, longitude,
        time, and value columns (column names configurable below).
    samples : pd.DataFrame
        Grid samples with the same required columns (sample-specific column
        names can be configured with the ``sample_*`` arguments).
    distance_tolerance_km : float
        Maximum allowed great-circle distance of a feasible pair, in
        kilometers. Must be non-negative.
    time_tolerance : pd.Timedelta or str
        Maximum allowed absolute time difference of a feasible pair. Strings
        are parsed by :class:`pandas.Timedelta` (e.g. ``"6h"``).
    period_freq : str or None, optional
        Pandas frequency alias (e.g. ``"D"``, ``"ME"``) used to derive each
        record's product period by rounding its timestamp into the containing
        period (daily/hourly aliases floor to the start; monthly and other
        end-of-period aliases label the whole period). Ignored when
        ``period_column`` is given. Exactly one of ``period_freq`` or
        ``period_column`` must be provided.
    period_column : str or None, optional
        Name of an existing column in both frames holding explicit product
        period labels. When given, ``period_freq`` is ignored.
    station_id_col, sample_id_col : str, optional
        Identifier column names in ``observations`` and ``samples``.
    lat_col, lon_col, time_col, value_col : str, optional
        Column names in ``observations`` for latitude, longitude, timestamp,
        and measured value.
    sample_lat_col, sample_lon_col, sample_time_col, sample_value_col : str or None, optional
        Column names in ``samples``. Default to the corresponding
        observation-side names when ``None``.

    Returns
    -------
    tuple of pd.DataFrame
        A four-tuple ``(matches, unmatched_observations, unmatched_samples,
        bias_summary)``:

        * ``matches``: one row per selected pair with identifiers, period,
          great-circle distance in kilometers, actual time difference, both
          values, and ``bias`` (sample value minus observation value).
        * ``unmatched_observations``: every unpaired observation with a
          rejection ``reason`` and its actual time difference to the closest
          candidate in-period (``NaT`` when no candidate exists).
        * ``unmatched_samples``: every unpaired sample with a rejection
          ``reason`` and the actual time difference to its closest feasible
          observation (``NaT`` when none exists).
        * ``bias_summary``: a single-row summary computed only from selected
          pairs, with pair count, mean bias, mean absolute error, root mean
          square error, mean distance, and mean absolute time difference.
          Pairs whose value is missing on either side are excluded from the
          value-based statistics.

    Raises
    ------
    ValueError
        If inputs are missing required columns, identifiers are duplicated,
        latitudes are out of range, tolerances are invalid, or neither / both
        period specifications are provided.
    """
    obs, smp, time_tol = _validate_and_prepare(
        observations,
        samples,
        distance_tolerance_km=distance_tolerance_km,
        time_tolerance=time_tolerance,
        period_freq=period_freq,
        period_column=period_column,
        station_id_col=station_id_col,
        sample_id_col=sample_id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        value_col=value_col,
        sample_lat_col=sample_lat_col,
        sample_lon_col=sample_lon_col,
        sample_time_col=sample_time_col,
        sample_value_col=sample_value_col,
    )

    # Stable ranks from the sorted identifiers make every tie-break invariant
    # to input row order. The type-aware key tolerates identifiers whose
    # inferred dtype is object (e.g. mixed integers and strings).
    obs = obs.sort_values("_id", key=_stable_sort_key).reset_index(drop=True)
    smp = smp.sort_values("_id", key=_stable_sort_key).reset_index(drop=True)
    obs["_rank"] = np.arange(len(obs), dtype=np.int64)
    smp["_rank"] = np.arange(len(smp), dtype=np.int64)

    matched_rows: list[dict[str, Any]] = []
    unmatched_obs_rows: list[dict[str, Any]] = []
    unmatched_smp_rows: list[dict[str, Any]] = []

    obs_used = np.zeros(len(obs), dtype=bool)
    smp_used = np.zeros(len(smp), dtype=bool)

    periods = pd.Index(
        sorted(set(obs["_period"].unique()).union(smp["_period"].unique())),
        name="period",
    )

    for period in periods:
        o_idx = np.flatnonzero(obs["_period"].to_numpy() == period)
        s_idx = np.flatnonzero(smp["_period"].to_numpy() == period)
        period_matches, period_uo, period_us = _process_period(
            obs.iloc[o_idx],
            smp.iloc[s_idx],
            period,
            distance_tolerance_km,
            time_tol,
        )
        matched_rows.extend(period_matches)
        unmatched_obs_rows.extend(period_uo)
        unmatched_smp_rows.extend(period_us)
        obs_used[o_idx] = True
        smp_used[s_idx] = True

    # Defensive backstop: every record must be accounted for exactly once.
    assert obs_used.all() and smp_used.all()

    matches = pd.DataFrame(matched_rows, columns=_MATCHED_COLUMNS)
    unmatched_obs_df = pd.DataFrame(
        unmatched_obs_rows, columns=_UNMATCHED_STATION_COLUMNS
    )
    unmatched_smp_df = pd.DataFrame(
        unmatched_smp_rows, columns=_UNMATCHED_SAMPLE_COLUMNS
    )
    bias_summary = _summarize_bias(matches)

    logger.info(
        "Matched %s of %s observations against %s samples across %s periods.",
        len(matches),
        len(obs),
        len(smp),
        len(periods),
    )
    return matches, unmatched_obs_df, unmatched_smp_df, bias_summary


def _stable_sort_key(values: pd.Series) -> pd.Series:
    """Ordering key for identifiers that may have mixed or object dtypes.

    Homogeneous numeric identifiers sort numerically; everything else is
    stringified first so heterogeneous identifiers (e.g. ``1`` and ``"s1"``)
    never raise ``TypeError: '<' not supported`` during sorting.
    """
    if pd.api.types.is_numeric_dtype(values):
        return values
    return values.map(lambda v: ("", v) if isinstance(v, str) else (type(v).__name__, str(v)))


def _validate_and_prepare(
    observations: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    distance_tolerance_km: float,
    time_tolerance: pd.Timedelta | str,
    period_freq: str | None,
    period_column: str | None,
    station_id_col: str,
    sample_id_col: str,
    lat_col: str,
    lon_col: str,
    time_col: str,
    value_col: str,
    sample_lat_col: str | None,
    sample_lon_col: str | None,
    sample_time_col: str | None,
    sample_value_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timedelta]:
    """Validate arguments and normalize both input frames."""
    if distance_tolerance_km < 0.0:
        raise ValueError("'distance_tolerance_km' must be non-negative.")
    parsed_time_tol = pd.Timedelta(time_tolerance)
    if parsed_time_tol < pd.Timedelta(0):
        raise ValueError("'time_tolerance' must be non-negative.")
    if (period_freq is None) == (period_column is None):
        raise ValueError(
            "Provide exactly one of 'period_freq' or 'period_column'."
        )

    sample_lat_col = sample_lat_col or lat_col
    sample_lon_col = sample_lon_col or lon_col
    sample_time_col = sample_time_col or time_col
    sample_value_col = sample_value_col or value_col

    obs = _prepare_frame(
        observations,
        id_col=station_id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        value_col=value_col,
        period_freq=period_freq,
        period_column=period_column,
        frame_label="observations",
    )
    smp = _prepare_frame(
        samples,
        id_col=sample_id_col,
        lat_col=sample_lat_col,
        lon_col=sample_lon_col,
        time_col=sample_time_col,
        value_col=sample_value_col,
        period_freq=period_freq,
        period_column=period_column,
        frame_label="samples",
    )
    return obs, smp, parsed_time_tol


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    time_col: str,
    value_col: str,
    period_freq: str | None,
    period_column: str | None,
    frame_label: str,
) -> pd.DataFrame:
    """Validate an input frame and normalize it into internal columns."""
    required = [id_col, lat_col, lon_col, time_col, value_col]
    if period_column is not None:
        required.append(period_column)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in {frame_label}: {missing}."
        )

    if frame[id_col].duplicated().any():
        raise ValueError(
            f"Identifiers in '{id_col}' ({frame_label}) must be unique."
        )

    lat = frame[lat_col].to_numpy(dtype=np.float64)
    if np.any(np.abs(lat) > 90.0):
        raise ValueError(
            f"Latitudes in '{lat_col}' ({frame_label}) must be within "
            "[-90, 90] degrees."
        )

    times = pd.to_datetime(frame[time_col])
    if times.isna().any():
        raise ValueError(
            f"Timestamps in '{time_col}' ({frame_label}) must all be valid."
        )
    if times.dt.tz is not None:
        # Normalize to tz-naive so cross-frame arithmetic is well defined.
        times = times.dt.tz_convert(None)

    if period_column is not None:
        periods = frame[period_column]
    else:
        assert period_freq is not None
        # Period-based grouping supports non-fixed frequencies such as "ME",
        # "QE", or "YE" (unlike ``DatetimeIndex.floor``). Period construction
        # only accepts the legacy base aliases ("M", "Q", "Y"), so both the
        # start- and end-of-period aliases are normalized first.
        period_alias = period_freq
        for modern_alias, legacy_alias in (
            ("MS", "M"),
            ("ME", "M"),
            ("QS", "Q"),
            ("QE", "Q"),
            ("YS", "Y"),
            ("YE", "Y"),
        ):
            if period_freq.startswith(modern_alias):
                period_alias = period_freq.replace(modern_alias, legacy_alias)
        periods = pd.Series(times.dt.to_period(period_alias).astype(str))

    normalized = pd.DataFrame(
        {
            "_id": frame[id_col].to_numpy(),
            "_lat": lat,
            "_lon": frame[lon_col].to_numpy(dtype=np.float64),
            "_time": times.to_numpy(),
            "_value": frame[value_col].to_numpy(dtype=np.float64),
            "_period": periods.to_numpy(),
        }
    )
    return normalized


def _process_period(
    obs: pd.DataFrame,
    smp: pd.DataFrame,
    period: Any,
    distance_tolerance_km: float,
    time_tol: pd.Timedelta,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce all match/rejection rows for a single product period."""
    matched: list[dict[str, Any]] = []
    unmatched_obs: list[dict[str, Any]] = []
    unmatched_smp: list[dict[str, Any]] = []

    if len(obs) == 0:
        for j in range(len(smp)):
            unmatched_smp.append(
                _unmatched_sample_row(smp.iloc[j], period, REASON_PERIOD, pd.NaT)
            )
        return matched, unmatched_obs, unmatched_smp
    if len(smp) == 0:
        for i in range(len(obs)):
            unmatched_obs.append(
                _unmatched_station_row(
                    obs.iloc[i], period, REASON_PERIOD, pd.NaT
                )
            )
        return matched, unmatched_obs, unmatched_smp

    result = _match_period(obs, smp, distance_tolerance_km, time_tol)
    for r, c in result["pairs"]:
        matched.append(
            _matched_row(
                obs.iloc[r],
                smp.iloc[c],
                period,
                result["dist"][r, c],
                result["dt"][r, c],
            )
        )
    for r in result["unmatched_obs"]:
        reason, closest_dt = _reject_station(
            result["dist"][r], result["dt"][r], distance_tolerance_km, time_tol
        )
        unmatched_obs.append(
            _unmatched_station_row(obs.iloc[r], period, reason, closest_dt)
        )
    for c in result["unmatched_smp"]:
        reason, closest_dt = _reject_sample(
            result["dist"][:, c],
            result["dt"][:, c],
            distance_tolerance_km,
            time_tol,
        )
        unmatched_smp.append(
            _unmatched_sample_row(smp.iloc[c], period, reason, closest_dt)
        )
    return matched, unmatched_obs, unmatched_smp


def _match_period(
    obs: pd.DataFrame,
    smp: pd.DataFrame,
    distance_tolerance_km: float,
    time_tol: pd.Timedelta,
) -> dict[str, Any]:
    """Solve the one-to-one assignment within a single product period."""
    cost, dist, dt, feasible = _build_cost_matrix(
        obs, smp, distance_tolerance_km, time_tol
    )
    row_pos, col_pos = linear_sum_assignment(cost)

    pairs: list[tuple[int, int]] = []
    paired_o: set[int] = set()
    paired_s: set[int] = set()
    for r, c in zip(row_pos, col_pos, strict=True):
        if not feasible[r, c]:
            continue
        pairs.append((r, c))
        paired_o.add(r)
        paired_s.add(c)

    return {
        "pairs": pairs,
        "unmatched_obs": [r for r in range(len(obs)) if r not in paired_o],
        "unmatched_smp": [c for c in range(len(smp)) if c not in paired_s],
        "dist": dist,
        "dt": dt,
    }


def _build_cost_matrix(
    obs: pd.DataFrame,
    smp: pd.DataFrame,
    distance_tolerance_km: float,
    time_tol: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the lexicographic assignment cost matrix for one period.

    Every feasible edge cost encodes four strict priority levels in one
    scalar::

        cost = great-circle distance (km) + identity tie-breaker (< epsilon)

    * Infeasible edges carry a constant penalty above every possible feasible
      assignment total, so the assignment first maximizes the number of
      feasible pairs.
    * The distance component then minimizes total great-circle distance.
    * The identity component (sample rank before observation rank) only
      changes the ordering when distance sums are equal to sub-millimeter
      precision; exactly equal distances are always resolved by stable
      identity rather than input row order.

    Returns
    -------
    tuple
        ``(cost, distance, time_delta, feasible)`` where ``feasible`` is the
        boolean tolerance mask.
    """
    n_o, n_s = len(obs), len(smp)
    if max(n_o, n_s) > _MAX_PERIOD_RECORDS:
        raise ValueError(
            "A single product period contains more than "
            f"{_MAX_PERIOD_RECORDS} records; split the inputs into smaller "
            "periods to keep tie-break costs numerically well separated."
        )

    dist = great_circle_distance_km(
        obs["_lat"].to_numpy()[:, None],
        obs["_lon"].to_numpy()[:, None],
        smp["_lat"].to_numpy()[None, :],
        smp["_lon"].to_numpy()[None, :],
    )

    dt_ns = np.abs(
        obs["_time"].to_numpy(dtype="datetime64[ns]")[:, None]
        - smp["_time"].to_numpy(dtype="datetime64[ns]")[None, :]
    )
    dt = pd.to_timedelta(dt_ns.flatten()).to_numpy().reshape(n_o, n_s)

    feasible = (dist <= distance_tolerance_km) & (dt_ns <= time_tol.to_numpy())

    obs_ranks = obs["_rank"].to_numpy(dtype=np.float64)[:, None]
    smp_ranks = smp["_rank"].to_numpy(dtype=np.float64)[None, :]
    max_rank = max(n_o, n_s)
    # Sample identity outranks observation identity: sample rank spans the
    # whole epsilon range while the observation contribution stays below one
    # unit of sample rank. Both terms combined stay strictly below epsilon.
    identity = _IDENTITY_EPSILON * (
        smp_ranks + obs_ranks / max(max_rank, 1)
    ) / max(max_rank, 1)

    # A rectangular assignment uses at most max(n_o, n_s) edges, so any
    # feasible total is below this bound. The infeasible penalty sits a full
    # factor above it, keeping feasibility and the distance objectives
    # numerically separated.
    max_feasible_total = (
        distance_tolerance_km * max_rank + _IDENTITY_EPSILON + 1.0
    )
    cost = np.where(
        feasible,
        dist + identity,
        _INFEASIBLE_FACTOR * max_feasible_total,
    )
    return cost, dist, dt, feasible


def _reject_station(
    dist_row: np.ndarray,
    dt_row: np.ndarray,
    distance_tolerance_km: float,
    time_tol: pd.Timedelta,
) -> tuple[str, pd.Timedelta]:
    """Determine why an observation was left unpaired.

    Only called for periods that contain at least one sample.
    """
    dt_ns = pd.to_timedelta(dt_row).to_numpy()
    closest = int(np.argmin(dt_ns))
    closest_dt = pd.Timedelta(dt_row[closest])

    if not (dist_row <= distance_tolerance_km).any():
        return REASON_DISTANCE, closest_dt
    if not (dt_ns <= time_tol.to_numpy()).any():
        return REASON_TIME, closest_dt
    return REASON_ASSIGNMENT, closest_dt


def _reject_sample(
    dist_col: np.ndarray,
    dt_col: np.ndarray,
    distance_tolerance_km: float,
    time_tol: pd.Timedelta,
) -> tuple[str, pd.Timedelta]:
    """Determine why a grid sample was left unpaired.

    Only called for periods that contain at least one observation.
    """
    dt_ns = pd.to_timedelta(dt_col).to_numpy()
    closest = int(np.argmin(dt_ns))
    closest_dt = pd.Timedelta(dt_col[closest])

    if not (dist_col <= distance_tolerance_km).any():
        return REASON_DISTANCE, closest_dt
    if not (dt_ns <= time_tol.to_numpy()).any():
        return REASON_TIME, closest_dt
    return REASON_ASSIGNMENT, closest_dt


def _matched_row(
    obs_row: pd.Series,
    smp_row: pd.Series,
    period: Any,
    distance_km: float,
    time_delta: np.timedelta64,
) -> dict[str, Any]:
    obs_val = obs_row["_value"]
    smp_val = smp_row["_value"]
    bias = (
        smp_val - obs_val
        if not (np.isnan(obs_val) or np.isnan(smp_val))
        else np.nan
    )
    return {
        "station_id": obs_row["_id"],
        "sample_id": smp_row["_id"],
        "period": period,
        "distance_km": distance_km,
        "time_delta": pd.Timedelta(time_delta),
        "observation_value": obs_val,
        "sample_value": smp_val,
        "bias": bias,
    }


def _unmatched_station_row(
    obs_row: pd.Series, period: Any, reason: str, time_delta: pd.Timedelta
) -> dict[str, Any]:
    return {
        "station_id": obs_row["_id"],
        "period": period,
        "reason": reason,
        "time_delta": time_delta,
    }


def _unmatched_sample_row(
    smp_row: pd.Series, period: Any, reason: str, time_delta: pd.Timedelta
) -> dict[str, Any]:
    return {
        "sample_id": smp_row["_id"],
        "period": period,
        "reason": reason,
        "time_delta": time_delta,
    }


def _summarize_bias(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute bias statistics using only selected pairs."""
    n_pairs = len(matches)
    if n_pairs == 0:
        return pd.DataFrame(
            [
                {
                    "n_pairs": 0,
                    "n_value_pairs": 0,
                    "bias_mean": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mean_distance_km": np.nan,
                    "mean_abs_time_delta": pd.NaT,
                }
            ]
        )

    bias = matches["bias"].to_numpy(dtype=np.float64)
    valid = ~np.isnan(bias)
    bias_v = bias[valid]
    return pd.DataFrame(
        [
            {
                "n_pairs": n_pairs,
                "n_value_pairs": int(valid.sum()),
                "bias_mean": float(np.mean(bias_v)) if valid.any() else np.nan,
                "mae": float(np.mean(np.abs(bias_v)))
                if valid.any()
                else np.nan,
                "rmse": float(np.sqrt(np.mean(bias_v**2)))
                if valid.any()
                else np.nan,
                "mean_distance_km": float(matches["distance_km"].mean()),
                "mean_abs_time_delta": pd.Timedelta(
                    matches["time_delta"].abs().mean()
                ),
            }
        ]
    )
