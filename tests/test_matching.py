"""Tests for one-to-one station-to-sample matching (xcdat.matching)."""

import numpy as np
import pandas as pd
import pytest

from xcdat import matching
from xcdat.matching import (
    REASON_ASSIGNMENT,
    REASON_DISTANCE,
    REASON_PERIOD,
    REASON_TIME,
    great_circle_distance_km,
    match_stations_to_samples,
)

KM_PER_DEGREE = 6371.0088 * np.pi / 180.0


def _obs(ids, lats, lons, times, values=None):
    return pd.DataFrame(
        {
            "station_id": list(ids),
            "lat": list(lats),
            "lon": list(lons),
            "time": pd.to_datetime(list(times)),
            "value": list(values if values is not None else np.zeros(len(ids))),
        }
    )


def _smp(ids, lats, lons, times, values=None):
    return pd.DataFrame(
        {
            "sample_id": list(ids),
            "lat": list(lats),
            "lon": list(lons),
            "time": pd.to_datetime(list(times)),
            "value": list(values if values is not None else np.ones(len(ids))),
        }
    )


class TestGreatCircleDistance:
    def test_zero_distance_same_point(self):
        np.testing.assert_allclose(
            great_circle_distance_km(10.0, 20.0, 10.0, 20.0), 0.0, atol=1e-12
        )

    def test_known_north_south_distance(self):
        # One degree of latitude is ~111.195 km.
        d = great_circle_distance_km(0.0, 0.0, 1.0, 0.0)
        np.testing.assert_allclose(d, KM_PER_DEGREE, rtol=1e-10)

    def test_equator_longitude_distance(self):
        d = great_circle_distance_km(0.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(d, KM_PER_DEGREE, rtol=1e-10)

    def test_antimeridian_equivalence(self):
        # 179.9 E and 179.9 W are only 0.2 degree apart.
        across = great_circle_distance_km(0.0, 179.9, 0.0, -179.9)
        direct = great_circle_distance_km(0.0, 0.0, 0.0, 0.2)
        np.testing.assert_allclose(across, direct, rtol=1e-12)

    def test_longitude_wrapping_invariant(self):
        # Longitudes outside [-180, 180] give the same answer after wrapping.
        d1 = great_circle_distance_km(45.0, -180.1, 45.0, 179.9)
        d2 = great_circle_distance_km(45.0, 179.9, 45.0, 179.9)
        np.testing.assert_allclose(d1, d2, rtol=1e-12)

    def test_vectorized_shapes(self):
        d = great_circle_distance_km(
            np.array([0.0, 10.0]),
            np.array([0.0, 0.0]),
            np.array([[0.0], [10.0]]),
            np.array([[0.0], [0.0]]),
        )
        assert d.shape == (2, 2)

    def test_invalid_latitude_raises(self):
        with pytest.raises(ValueError):
            great_circle_distance_km(90.1, 0.0, 0.0, 0.0)


class TestMatching:
    def test_basic_one_to_one_pair_with_bias(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"], [1.0])
        smp = _smp(["s1"], [0.0], [0.0], ["2020-01-01 01:00"], [1.5])

        m, uo, us, summary = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )

        assert len(m) == 1
        row = m.iloc[0]
        assert row["station_id"] == "A"
        assert row["sample_id"] == "s1"
        assert row["distance_km"] == pytest.approx(0.0, abs=1e-12)
        assert row["time_delta"] == pd.Timedelta("1h")
        assert row["bias"] == pytest.approx(0.5)
        assert uo.empty and us.empty
        assert summary.loc[0, "n_pairs"] == 1
        assert summary.loc[0, "bias_mean"] == pytest.approx(0.5)
        assert summary.loc[0, "mae"] == pytest.approx(0.5)
        assert summary.loc[0, "rmse"] == pytest.approx(0.5)
        assert summary.loc[0, "mean_abs_time_delta"] == pd.Timedelta("1h")

    def test_no_measurement_used_twice_same_period(self):
        # One sample close to two observations: only one pair may form.
        obs = _obs(
            ["A", "B"], [0.0, 0.01], [0.0, 0.0],
            ["2020-01-01", "2020-01-01"],
        )
        smp = _smp(["s1"], [0.0], [0.0], ["2020-01-01"])

        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )

        assert len(m) == 1
        assert m["station_id"].is_unique and m["sample_id"].is_unique
        assert len(uo) == 1
        assert uo.iloc[0]["reason"] == REASON_ASSIGNMENT
        assert us.empty

    def test_cardinality_outranks_total_distance(self):
        # Two observations, one sample. Greedy nearest pairing would take B
        # (closest to s1); the cardinality-maximizing assignment is identical
        # here in count, so instead build a 2x2 case where the distance-greedy
        # choice strands a record: s1 only reachable by A (B is out of range
        # of s1). Pairing A-s2 first would leave B with no feasible sample.
        obs = _obs(
            ["A", "B"], [0.0, 0.0], [0.0, 2.0],
            ["2020-01-01", "2020-01-01"],
        )
        smp = _smp(
            # s1 ~111 km from A and far from B; s2 ~111 km from both.
            ["s1", "s2"], [0.0, 0.0], [1.0, 1.0],
            ["2020-01-01", "2020-01-01"],
        )

        m, _, _, _ = match_stations_to_samples(
            obs, smp, 150.0, "6h", period_freq="D"
        )

        # Two pairs must be formed; that forces A-s1 and B-s2 even though
        # A-s1 and A-s2 are equally near A.
        assert len(m) == 2
        pairs = set(zip(m["station_id"], m["sample_id"], strict=True))
        assert pairs == {("A", "s1"), ("B", "s2")}

    def test_minimize_total_distance_among_maximal_matchings(self):
        # Both perfect matchings give two pairs; the lower total distance wins.
        obs = _obs(
            ["A", "B"], [0.0, 0.0], [0.0, 1.0],
            ["2020-01-01", "2020-01-01"],
        )
        smp = _smp(
            ["s1", "s2"], [0.0, 0.0], [0.05, 0.95],
            ["2020-01-01", "2020-01-01"],
        )

        m, _, _, _ = match_stations_to_samples(
            obs, smp, 500.0, "6h", period_freq="D"
        )
        pairs = set(zip(m["station_id"], m["sample_id"], strict=True))
        assert pairs == {("A", "s1"), ("B", "s2")}

    def test_tie_break_by_stable_sample_identity(self):
        # Two equidistant, equitemporal samples: lower sample_id wins,
        # regardless of input row order.
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"])
        smp = _smp(
            ["s_b", "s_a"], [0.0, 0.0], [0.0, 0.0],
            ["2020-01-01", "2020-01-01"],
        )

        m, _, us, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert m.iloc[0]["sample_id"] == "s_a"
        assert us.iloc[0]["sample_id"] == "s_b"
        assert us.iloc[0]["reason"] == REASON_ASSIGNMENT

        m2, _, _, _ = match_stations_to_samples(
            obs.iloc[::-1].reset_index(drop=True),
            smp.iloc[::-1].reset_index(drop=True),
            100.0,
            "6h",
            period_freq="D",
        )
        assert m2.iloc[0]["sample_id"] == "s_a"

    def test_distance_dominates_identity_tie_breaker(self):
        # The lexicographically lower sample id is farther away; a strictly
        # closer but higher-id sample must still be selected.
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"])
        smp = _smp(
            ["s_a", "s_b"], [0.0, 0.0], [0.20, 0.10],
            ["2020-01-01", "2020-01-01"],
        )

        m, _, _, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert m.iloc[0]["sample_id"] == "s_b"

    def test_shuffled_inputs_give_identical_result(self):
        rng = np.random.default_rng(42)
        obs = _obs(
            [f"A{i}" for i in range(6)],
            rng.uniform(-60, 60, 6),
            rng.uniform(-180, 180, 6),
            pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 20, 6), "h"),
            rng.normal(size=6),
        )
        smp = _smp(
            [f"s{i}" for i in range(6)],
            obs["lat"] + rng.uniform(-0.2, 0.2, 6),
            obs["lon"] + rng.uniform(-0.2, 0.2, 6),
            pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 20, 6), "h"),
            rng.normal(size=6),
        )

        m1, uo1, us1, s1 = match_stations_to_samples(
            obs, smp, 200.0, "12h", period_freq="D"
        )
        m2, uo2, us2, s2 = match_stations_to_samples(
            obs.sample(frac=1, random_state=7).reset_index(drop=True),
            smp.sample(frac=1, random_state=11).reset_index(drop=True),
            200.0,
            "12h",
            period_freq="D",
        )

        cols = ["station_id", "sample_id"]
        pd.testing.assert_frame_equal(
            m1.sort_values(cols).reset_index(drop=True),
            m2.sort_values(cols).reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            uo1.sort_values("station_id").reset_index(drop=True),
            uo2.sort_values("station_id").reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            us1.sort_values("sample_id").reset_index(drop=True),
            us2.sort_values("sample_id").reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(s1, s2)

    def test_antimeridian_pairing(self):
        obs = _obs(["X"], [0.0], [179.9], ["2020-01-01"])
        smp = _smp(["g"], [0.0], [-179.9], ["2020-01-01"])

        m, uo, _, _ = match_stations_to_samples(
            obs, smp, 50.0, "1h", period_freq="D"
        )
        assert len(m) == 1
        assert m.iloc[0]["distance_km"] == pytest.approx(
            0.2 * KM_PER_DEGREE, rel=1e-6
        )
        assert uo.empty

    def test_distance_tolerance_rejection_records_reason(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"])
        smp = _smp(["far"], [5.0], [0.0], ["2020-01-01"])  # ~556 km away

        m, uo, us, summary = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert m.empty
        assert uo.iloc[0]["reason"] == REASON_DISTANCE
        assert us.iloc[0]["reason"] == REASON_DISTANCE
        # Actual time difference is retained even for rejections.
        assert uo.iloc[0]["time_delta"] == pd.Timedelta(0)
        assert summary.loc[0, "n_pairs"] == 0
        assert np.isnan(summary.loc[0, "bias_mean"])

    def test_time_tolerance_rejection_records_actual_delta(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01 00:00"])
        smp = _smp(["late"], [0.0], [0.0], ["2020-01-01 12:00"])

        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert m.empty
        assert uo.iloc[0]["reason"] == REASON_TIME
        assert uo.iloc[0]["time_delta"] == pd.Timedelta("12h")
        assert us.iloc[0]["reason"] == REASON_TIME
        assert us.iloc[0]["time_delta"] == pd.Timedelta("12h")

    def test_periods_are_matched_independently(self):
        # Same sample id reused across two periods is allowed once per period.
        obs = _obs(
            ["A", "A2"],
            [0.0, 0.0],
            [0.0, 0.0],
            ["2020-01-01", "2020-01-02"],
        )
        smp = _smp(
            ["g", "g2"],
            [0.0, 0.0],
            [0.0, 0.0],
            ["2020-01-01", "2020-01-02"],
        )

        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert len(m) == 2
        assert set(m["period"]) == {"2020-01-01", "2020-01-02"}
        assert uo.empty and us.empty

    def test_no_sample_in_period_reason(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-15"])
        smp = _smp(
            ["g"], [0.0], [0.0], ["2020-02-15"],
        )
        obs["cycle"] = ["2020-01"]
        smp["cycle"] = ["2020-02"]

        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_column="cycle"
        )
        assert m.empty
        assert uo.iloc[0]["reason"] == REASON_PERIOD
        assert pd.isna(uo.iloc[0]["time_delta"])
        assert us.iloc[0]["reason"] == REASON_PERIOD

    def test_monthly_non_fixed_period_frequency(self):
        # "ME" is a non-fixed frequency that dt.floor() cannot handle;
        # period-based grouping keeps observations in separate monthly cycles.
        obs = _obs(
            ["A", "B"], [0.0, 0.0], [0.0, 0.0],
            ["2020-01-31", "2020-02-01"],
        )
        smp = _smp(
            ["g1", "g2"], [0.0, 0.0], [0.0, 0.0],
            ["2020-01-31", "2020-02-01"],
        )

        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="ME"
        )
        assert len(m) == 2
        assert set(m["period"]) == {"2020-01", "2020-02"}
        assert uo.empty and us.empty

    def test_start_of_period_aliases_are_normalized(self):
        # "MS" is not a valid Period alias; it must map to "M" internally.
        obs = _obs(
            ["A", "B"], [0.0, 0.0], [0.0, 0.0],
            ["2020-01-02", "2020-02-02"],
        )
        smp = _smp(
            ["g1", "g2"], [0.0, 0.0], [0.0, 0.0],
            ["2020-01-03", "2020-02-03"],
        )
        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 100.0, "48h", period_freq="MS"
        )
        assert len(m) == 2
        assert set(m["period"]) == {"2020-01", "2020-02"}
        assert uo.empty and us.empty

    def test_explicit_period_column(self):
        obs = _obs(["A", "B"], [0.0, 0.0], [0.0, 0.0],
                   ["2020-01-01", "2020-01-05"])
        smp = _smp(["g1", "g2"], [0.0, 0.0], [0.0, 0.0],
                   ["2020-01-02", "2020-01-06"])
        obs["cycle"] = ["P1", "P2"]
        smp["cycle"] = ["P1", "P2"]

        m, _, _, _ = match_stations_to_samples(
            obs,
            smp,
            100.0,
            "48h",
            period_column="cycle",
        )
        assert len(m) == 2
        assert set(m["period"]) == {"P1", "P2"}

    def test_every_record_accounted_for_once(self):
        rng = np.random.default_rng(3)
        obs = _obs(
            [f"A{i}" for i in range(5)],
            rng.uniform(-30, 30, 5),
            rng.uniform(-180, 180, 5),
            pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 48, 5), "h"),
        )
        smp = _smp(
            [f"s{i}" for i in range(7)],
            rng.uniform(-30, 30, 7),
            rng.uniform(-180, 180, 7),
            pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 48, 7), "h"),
        )

        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 150.0, "5h", period_freq="D"
        )
        assert len(m) + len(uo) == len(obs)
        assert len(m) + len(us) == len(smp)
        assert set(m["station_id"]).isdisjoint(set(uo["station_id"]))
        assert set(m["sample_id"]).isdisjoint(set(us["sample_id"]))

    def test_nan_values_excluded_from_bias_stats_but_pair_kept(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"], [np.nan])
        smp = _smp(["s1"], [0.0], [0.0], ["2020-01-01"], [1.5])

        m, _, _, summary = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert len(m) == 1
        assert np.isnan(m.iloc[0]["bias"])
        assert summary.loc[0, "n_pairs"] == 1
        assert summary.loc[0, "n_value_pairs"] == 0
        assert np.isnan(summary.loc[0, "rmse"])

    def test_rmse_and_mae_aggregation(self):
        obs = _obs(
            ["A", "B"], [0.0, 0.0], [0.0, 0.0],
            ["2020-01-01", "2020-01-01"], [0.0, 0.0],
        )
        smp = _smp(
            ["s1", "s2"], [0.0, 0.0], [0.0, 0.0],
            ["2020-01-01", "2020-01-01"], [1.0, -3.0],
        )
        _, _, _, summary = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert summary.loc[0, "bias_mean"] == pytest.approx(-1.0)
        assert summary.loc[0, "mae"] == pytest.approx(2.0)
        assert summary.loc[0, "rmse"] == pytest.approx(np.sqrt(5.0))

    def test_string_sample_column_names(self):
        obs = pd.DataFrame(
            {
                "stn": ["A"],
                "latitude": [0.0],
                "longitude": [0.0],
                "ts": pd.to_datetime(["2020-01-01"]),
                "meas": [1.0],
            }
        )
        smp = pd.DataFrame(
            {
                "grid": ["g"],
                "glat": [0.0],
                "glon": [0.0],
                "gts": pd.to_datetime(["2020-01-01"]),
                "gval": [2.0],
            }
        )
        m, _, _, _ = match_stations_to_samples(
            obs,
            smp,
            100.0,
            "6h",
            period_freq="D",
            station_id_col="stn",
            sample_id_col="grid",
            lat_col="latitude",
            lon_col="longitude",
            time_col="ts",
            value_col="meas",
            sample_lat_col="glat",
            sample_lon_col="glon",
            sample_time_col="gts",
            sample_value_col="gval",
        )
        assert m.iloc[0]["station_id"] == "A"
        assert m.iloc[0]["sample_id"] == "g"
        assert m.iloc[0]["bias"] == pytest.approx(1.0)

    def test_mixed_type_identifiers_do_not_raise(self):
        obs = pd.DataFrame(
            {
                "station_id": [2, "10", 1],
                "lat": [0.0, 0.0, 0.0],
                "lon": [0.0, 0.0, 0.0],
                "time": pd.to_datetime(["2020-01-01"] * 3),
                "value": [0.0, 0.0, 0.0],
            }
        )
        smp = pd.DataFrame(
            {
                "sample_id": ["b", 1, "a"],
                "lat": [0.0, 0.0, 0.0],
                "lon": [0.0, 0.0, 0.0],
                "time": pd.to_datetime(["2020-01-01"] * 3),
                "value": [1.0, 1.0, 1.0],
            }
        )
        m1, _, _, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        m2, _, _, _ = match_stations_to_samples(
            obs.iloc[[2, 0, 1]].reset_index(drop=True),
            smp.iloc[[1, 2, 0]].reset_index(drop=True),
            100.0,
            "6h",
            period_freq="D",
        )
        key = lambda frame: sorted(  # noqa: E731
            zip(frame["station_id"], frame["sample_id"]),
            key=lambda t: (str(t[0]), str(t[1])),
        )
        assert key(m1) == key(m2)
        assert len(m1) == 3

    def test_distance_and_time_reasons_coexist_in_one_period(self):
        # One observation fails the distance tolerance, another the time
        # tolerance, against the same in-period candidate set.
        obs = _obs(
            ["far", "late"],
            [0.0, 0.0],
            [0.0, 0.0],
            ["2020-01-01 00:00", "2020-01-01 00:00"],
        )
        smp = pd.DataFrame(
            {
                "sample_id": ["near"],
                "lat": [5.0],  # ~556 km from both observations
                "lon": [0.0],
                "time": pd.to_datetime(["2020-01-01 12:00"]),
                "value": [1.0],
            }
        )

        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert m.empty
        reasons = dict(zip(uo["station_id"], uo["reason"], strict=True))
        # Distance failure takes precedence in the reason classification.
        assert reasons == {"far": REASON_DISTANCE, "late": REASON_DISTANCE}
        # Actual time differences are still recorded for both rejections.
        assert set(uo["time_delta"]) == {pd.Timedelta("12h")}

        # Widen the distance tolerance: now only the time failure remains.
        m, uo, us, _ = match_stations_to_samples(
            obs, smp, 1000.0, "6h", period_freq="D"
        )
        assert m.empty
        reasons = dict(zip(uo["station_id"], uo["reason"], strict=True))
        assert set(reasons.values()) == {REASON_TIME}
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"])
        smp_exact = _smp(["g"], [0.0], [0.0], ["2020-01-01"])
        smp_near = _smp(["g"], [1e-6], [0.0], ["2020-01-01"])

        m, uo, _, _ = match_stations_to_samples(
            obs, smp_exact, 0.0, "1h", period_freq="D"
        )
        assert len(m) == 1 and uo.empty

        m, uo, _, _ = match_stations_to_samples(
            obs, smp_near, 0.0, "1h", period_freq="D"
        )
        assert m.empty
        assert uo.iloc[0]["reason"] == REASON_DISTANCE

    def test_empty_input_frames(self):
        obs = _obs([], [], [], [])
        smp = _smp([], [], [], [])
        m, uo, us, summary = match_stations_to_samples(
            obs, smp, 100.0, "6h", period_freq="D"
        )
        assert m.empty and uo.empty and us.empty
        assert summary.loc[0, "n_pairs"] == 0
        assert np.isnan(summary.loc[0, "rmse"])

    def test_timedelta_dtype_preserved(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01 00:00"])
        smp = _smp(["g"], [0.0], [0.0], ["2020-01-01 00:30"])
        m, _, _, summary = match_stations_to_samples(
            obs, smp, 10.0, "1h", period_freq="D"
        )
        assert m["time_delta"].dtype == "timedelta64[ns]"
        assert summary.loc[0, "mean_abs_time_delta"] == pd.Timedelta("30min")

    def test_hourly_period_labels(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01 00:45"])
        smp = _smp(["g"], [0.0], [0.0], ["2020-01-01 00:10"])
        m, _, _, _ = match_stations_to_samples(
            obs, smp, 10.0, "1h", period_freq="h"
        )
        assert len(m) == 1
        assert m.iloc[0]["period"] == "2020-01-01 00:00"


class TestValidation:
    def test_missing_column_raises(self):
        obs = pd.DataFrame({"station_id": ["A"], "lat": [0.0]})
        smp = _smp(["g"], [0.0], [0.0], ["2020-01-01"])
        with pytest.raises(ValueError, match="Missing required column"):
            match_stations_to_samples(
                obs, smp, 100.0, "6h", period_freq="D"
            )

    def test_duplicate_identifiers_raise(self):
        obs = _obs(
            ["A", "A"], [0.0, 1.0], [0.0, 1.0],
            ["2020-01-01", "2020-01-01"],
        )
        smp = _smp(["g"], [0.0], [0.0], ["2020-01-01"])
        with pytest.raises(ValueError, match="must be unique"):
            match_stations_to_samples(
                obs, smp, 100.0, "6h", period_freq="D"
            )

    def test_negative_tolerances_raise(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"])
        smp = _smp(["g"], [0.0], [0.0], ["2020-01-01"])
        with pytest.raises(ValueError, match="distance"):
            match_stations_to_samples(obs, smp, -1.0, "6h", period_freq="D")
        with pytest.raises(ValueError, match="time"):
            match_stations_to_samples(obs, smp, 100.0, "-1h", period_freq="D")

    def test_period_spec_exclusive(self):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"])
        smp = _smp(["g"], [0.0], [0.0], ["2020-01-01"])
        with pytest.raises(ValueError, match="exactly one"):
            match_stations_to_samples(obs, smp, 100.0, "6h")
        with pytest.raises(ValueError, match="exactly one"):
            match_stations_to_samples(
                obs, smp, 100.0, "6h", period_freq="D", period_column="p"
            )

    def test_invalid_latitude_raises(self):
        obs = _obs(["A"], [91.0], [0.0], ["2020-01-01"])
        smp = _smp(["g"], [0.0], [0.0], ["2020-01-01"])
        with pytest.raises(ValueError, match="[Ll]atitude"):
            match_stations_to_samples(
                obs, smp, 100.0, "6h", period_freq="D"
            )

    def test_reason_constants_exposed(self):
        # Rejection vocabulary is part of the public surface.
        assert matching.REASON_ASSIGNMENT
        assert matching.REASON_DISTANCE
        assert matching.REASON_TIME
        assert matching.REASON_PERIOD

    def test_period_size_guard_raises(self, monkeypatch):
        obs = _obs(["A"], [0.0], [0.0], ["2020-01-01"])
        smp = _smp(["g"], [0.0], [0.0], ["2020-01-01"])
        monkeypatch.setattr(matching, "_MAX_PERIOD_RECORDS", 0)
        with pytest.raises(ValueError, match="single product period contains"):
            match_stations_to_samples(
                obs, smp, 100.0, "6h", period_freq="D"
            )
