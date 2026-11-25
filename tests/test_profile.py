"""Tests for vertical profile restoration, merging, and interpolation."""

import numpy as np
import pytest

from xcdat.profile import (
    REASON_INTERNAL_GAP,
    REASON_NO_VALID_DATA,
    REASON_OUT_OF_RANGE,
    Profile,
    ProfileMergeError,
    RestoredProfile,
    interpolate_profile,
    merge_and_interpolate,
    merge_profiles,
    restore_profile,
)


class TestRestoreProfile:
    def test_downward_metres_sorted_with_mask(self):
        # Depths intentionally out of order; values and mask must follow.
        profile = Profile(
            values=np.array([30.0, 10.0, 20.0]),
            depths=np.array([30.0, 10.0, 20.0]),
            mask=np.array([True, True, False]),
            units="m",
            positive="down",
            name="p1",
        )

        result = restore_profile(profile)

        assert isinstance(result, RestoredProfile)
        np.testing.assert_array_equal(result.depths, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(result.values, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(result.mask, [True, False, True])

    def test_upward_centimetres_flips_sign_and_converts_units(self):
        # Upward-positive cm: recorded [1000, 0, -1000] -> physical
        # depths [-10, 0, 10] m (positive downward), then sorted. The value
        # recorded at each level travels with its restored depth.
        profile = Profile(
            values=np.array([9.0, 2.0, 7.0]),
            depths=np.array([1000.0, 0.0, -1000.0]),
            units="cm",
            positive="up",
            name="p2",
        )

        result = restore_profile(profile)

        np.testing.assert_allclose(result.depths, [-10.0, 0.0, 10.0])
        # Restored depth -10 m came from recorded +1000 cm (value 9.0);
        # restored depth +10 m came from recorded -1000 cm (value 7.0);
        # values and mask travel with their restored depth.
        np.testing.assert_array_equal(result.values, [9.0, 2.0, 7.0])
        np.testing.assert_array_equal(result.mask, [True, True, True])

    def test_default_mask_treats_nan_as_invalid(self):
        profile = Profile(
            values=np.array([1.0, np.nan, 3.0]),
            depths=np.array([1.0, 2.0, 3.0]),
            units="metres",
            positive="downward",
        )

        result = restore_profile(profile)

        np.testing.assert_array_equal(result.mask, [True, False, True])

    def test_explicit_valid_mask_on_nan_value_is_dropped(self):
        profile = Profile(
            values=np.array([1.0, np.nan]),
            depths=np.array([1.0, 2.0]),
            mask=np.array([True, True]),
            units="m",
            positive="down",
        )

        result = restore_profile(profile)

        np.testing.assert_array_equal(result.mask, [True, False])

    def test_unknown_unit_and_direction_raise(self):
        with pytest.raises(ValueError, match="Unknown depth unit"):
            restore_profile(
                Profile(
                    values=[1.0],
                    depths=[1.0],
                    units="fathom",
                    positive="down",
                    name="bad-unit",
                )
            )

        with pytest.raises(ValueError, match="Unknown depth direction"):
            restore_profile(
                Profile(
                    values=[1.0],
                    depths=[1.0],
                    units="m",
                    positive="sideways",
                    name="bad-dir",
                )
            )

    def test_length_mismatch_and_nonfinite_depth_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            restore_profile(
                Profile(values=[1.0, 2.0], depths=[1.0], units="m", positive="down")
            )

        with pytest.raises(ValueError, match="finite"):
            restore_profile(
                Profile(
                    values=[1.0, 2.0],
                    depths=[1.0, np.nan],
                    units="m",
                    positive="down",
                )
            )


class TestMergeProfiles:
    def test_metres_downward_and_centimetres_upward_merge_on_physical_depth(self):
        # Profile A: metres, positive downward.
        profile_a = Profile(
            values=np.array([10.0, 20.0, 30.0]),
            depths=np.array([0.0, 10.0, 20.0]),
            units="m",
            positive="down",
            name="A",
        )
        # Profile B: centimetres, positive upward -> its levels -500cm and
        # -1500cm become physical depths 5 m and 15 m.
        profile_b = Profile(
            values=np.array([15.0, 25.0]),
            depths=np.array([-500.0, -1500.0]),
            units="cm",
            positive="up",
            name="B",
        )

        merged = merge_profiles([profile_a, profile_b])

        np.testing.assert_allclose(merged.depths, [0.0, 5.0, 10.0, 15.0, 20.0])
        np.testing.assert_allclose(merged.values, [10.0, 15.0, 20.0, 25.0, 30.0])
        assert merged.mask.all()

    def test_contradiction_across_profiles_raises_and_names_profiles(self):
        profile_a = Profile(
            values=[42.0], depths=[10.0], units="m", positive="down", name="A"
        )
        # 1000 cm upward-positive == 10 m downward-positive physical depth.
        profile_b = Profile(
            values=[99.0], depths=[-1000.0], units="cm", positive="up", name="B"
        )

        with pytest.raises(ProfileMergeError) as exc_info:
            merge_profiles([profile_a, profile_b])

        message = str(exc_info.value)
        assert "A" in message and "B" in message
        assert "42" in message and "99" in message

    def test_contradiction_within_single_profile_raises(self):
        profile = Profile(
            values=[1.0, 2.0],
            depths=[5.0, 5.0],
            units="m",
            positive="down",
            name="solo",
        )

        with pytest.raises(ProfileMergeError, match="solo"):
            merge_profiles([profile])

    def test_agreement_on_shared_depth_is_kept_once(self):
        profile_a = Profile(
            values=[7.0, 8.0], depths=[10.0, 20.0], units="m", positive="down",
            name="A",
        )
        profile_b = Profile(
            values=[7.0], depths=[-1000.0], units="cm", positive="up", name="B"
        )

        merged = merge_profiles([profile_a, profile_b])

        np.testing.assert_allclose(merged.depths, [10.0, 20.0])
        np.testing.assert_allclose(merged.values, [7.0, 8.0])

    def test_invalid_observations_do_not_trigger_conflict(self):
        profile_a = Profile(
            values=[42.0], depths=[10.0], units="m", positive="down", name="A"
        )
        profile_b = Profile(
            values=[np.nan],
            depths=[-1000.0],
            mask=np.array([False]),
            units="cm",
            positive="up",
            name="B",
        )

        merged = merge_profiles([profile_a, profile_b])

        np.testing.assert_allclose(merged.depths, [10.0])
        np.testing.assert_allclose(merged.values, [42.0])
        np.testing.assert_array_equal(merged.mask, [True])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="At least one profile"):
            merge_profiles([])


class TestInterpolateProfile:
    @pytest.fixture
    def restored(self):
        # Physical depths 0..40 m with a missing level at 20 m creating an
        # internal gap between the valid levels 10 m and 30 m.
        return RestoredProfile(
            name="merged",
            depths=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
            values=np.array([0.0, 100.0, np.nan, 300.0, 400.0]),
            mask=np.array([True, True, False, True, True]),
        )

    def test_linear_interpolation_between_valid_neighbours(self, restored):
        result = interpolate_profile(restored, np.array([5.0, 35.0]))

        np.testing.assert_allclose(result.values, [50.0, 350.0])
        assert result.mask.all()
        assert all(reason is None for reason in result.reasons)

    def test_exact_sample_depth_returns_sample_value(self, restored):
        result = interpolate_profile(restored, np.array([10.0]))

        assert result.values[0] == 100.0
        assert result.mask[0]

    def test_missing_level_depth_reports_internal_gap(self, restored):
        # 20 m itself is a missing level; it is bounded by valid levels 10 and
        # 30 whose span crosses the internal fault.
        result = interpolate_profile(restored, np.array([20.0]))

        assert np.isnan(result.values[0])
        assert not result.mask[0]
        assert result.reasons[0] == REASON_INTERNAL_GAP

    def test_target_inside_gap_interval_reports_internal_gap(self, restored):
        result = interpolate_profile(restored, np.array([15.0, 25.0]))

        assert not result.mask.any()
        assert list(result.reasons) == [REASON_INTERNAL_GAP, REASON_INTERNAL_GAP]

    def test_out_of_range_is_not_extrapolated(self, restored):
        targets = np.array([-5.0, 45.0])
        result = interpolate_profile(restored, targets)

        assert np.all(np.isnan(result.values))
        assert not result.mask.any()
        assert list(result.reasons) == [
            REASON_OUT_OF_RANGE,
            REASON_OUT_OF_RANGE,
        ]

    def test_edge_targets_at_valid_range_bounds_are_kept(self, restored):
        result = interpolate_profile(restored, np.array([0.0, 40.0]))

        np.testing.assert_allclose(result.values, [0.0, 400.0])
        assert result.mask.all()

    def test_no_valid_observations_reports_reason(self):
        restored = RestoredProfile(
            name="empty",
            depths=np.array([1.0, 2.0]),
            values=np.array([np.nan, np.nan]),
            mask=np.array([False, False]),
        )

        result = interpolate_profile(restored, np.array([1.0, 1.5]))

        assert not result.mask.any()
        assert list(result.reasons) == [
            REASON_NO_VALID_DATA,
            REASON_NO_VALID_DATA,
        ]

    def test_nonfinite_target_raises(self, restored):
        with pytest.raises(ValueError, match="finite"):
            interpolate_profile(restored, np.array([np.nan]))

    def test_multidimensional_target_raises(self, restored):
        with pytest.raises(ValueError, match="1-D"):
            interpolate_profile(restored, np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestMergeAndInterpolate:
    def test_end_to_end_mixed_units_and_directions(self):
        # A: metres downward-positive, valid 0 m and 10 m, gap at 20 m.
        profile_a = Profile(
            values=np.array([0.0, 100.0, np.nan]),
            depths=np.array([0.0, 10.0, 20.0]),
            mask=np.array([True, True, False]),
            units="m",
            positive="down",
            name="A",
        )
        # B: centimetres upward-positive; -3000 cm and -4000 cm -> 30 m and
        # 40 m physical depth.
        profile_b = Profile(
            values=np.array([300.0, 400.0]),
            depths=np.array([-3000.0, -4000.0]),
            units="cm",
            positive="up",
            name="B",
        )

        targets = np.array([5.0, 15.0, 25.0, 35.0, 50.0])
        result = merge_and_interpolate([profile_a, profile_b], targets)

        # 5 m: linear between 0 and 10. 35 m: linear between 30 and 40.
        assert result.mask.tolist() == [True, False, False, True, False]
        np.testing.assert_allclose(result.values[result.mask], [50.0, 350.0])
        # 15 m and 25 m straddle the internal gap; 50 m is out of range.
        assert result.reasons.tolist() == [
            None,
            REASON_INTERNAL_GAP,
            REASON_INTERNAL_GAP,
            None,
            REASON_OUT_OF_RANGE,
        ]

    def test_contradiction_surfaces_before_interpolation(self):
        profile_a = Profile(
            values=[1.0], depths=[10.0], units="m", positive="down", name="A"
        )
        profile_b = Profile(
            values=[2.0], depths=[-1000.0], units="cm", positive="up", name="B"
        )

        with pytest.raises(ProfileMergeError):
            merge_and_interpolate([profile_a, profile_b], np.array([10.0]))
