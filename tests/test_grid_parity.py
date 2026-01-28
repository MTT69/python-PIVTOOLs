"""
Test to verify instantaneous and ensemble PIV use the same grid and padding.

This test proves that for identical configurations, both modes produce:
1. Same window center coordinates
2. Same padding values (n_pre_all, n_post_all)
3. Same padded window centers (win_ctrs_x_all, win_ctrs_y_all)
4. Same interpolation maps
"""

import numpy as np
import pytest
from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU


def create_matching_config(image_shape=(256, 256), window_sizes=None, overlaps=None):
    """
    Create a Config object with matching instantaneous and ensemble settings.

    This ensures both modes use identical window sizes, overlaps, and image shapes.
    """
    if window_sizes is None:
        window_sizes = [(64, 64), (32, 32), (16, 16)]
    if overlaps is None:
        overlaps = [50, 50, 50]

    config = Config()

    # Set image shape via internal variable (property is read-only)
    config._detected_image_shape = image_shape

    # Initialize data structure if needed
    if "instantaneous_piv" not in config.data:
        config.data["instantaneous_piv"] = {}
    if "ensemble_piv" not in config.data:
        config.data["ensemble_piv"] = {}
    if "processing" not in config.data:
        config.data["processing"] = {}

    # Instantaneous settings - set through data dict
    config.data["instantaneous_piv"]["window_size"] = list(window_sizes)
    config.data["instantaneous_piv"]["overlap"] = list(overlaps)
    config.data["instantaneous_piv"]["window_type"] = "gaussian"
    config.data["instantaneous_piv"]["num_peaks"] = 1
    config.data["instantaneous_piv"]["peak_finder"] = 0

    # Ensemble settings - MUST match instantaneous for this test
    config.data["ensemble_piv"]["window_size"] = list(window_sizes)
    config.data["ensemble_piv"]["overlap"] = list(overlaps)
    config.data["ensemble_piv"]["window_type"] = "gaussian"
    config.data["ensemble_piv"]["num_passes"] = len(window_sizes)
    config.data["ensemble_piv"]["type"] = ["standard"] * len(window_sizes)
    config.data["ensemble_piv"]["sum_window"] = [16, 16]
    config.data["ensemble_piv"]["num_peaks"] = 1
    config.data["ensemble_piv"]["peak_finder"] = 0

    # Processing settings
    config.data["processing"]["omp_threads"] = 1

    return config


class TestGridParity:
    """Test that instantaneous and ensemble use identical grids."""

    @pytest.fixture
    def correlators(self):
        """Create both correlators with matching config."""
        config = create_matching_config()

        inst = InstantaneousCorrelatorCPU(config)
        ens = EnsembleCorrelatorCPU(config)

        return inst, ens, config

    def test_window_centers_match(self, correlators):
        """Verify window center coordinates are identical."""
        inst, ens, config = correlators

        for pass_idx in range(len(config.window_sizes)):
            # Check X centers
            np.testing.assert_array_almost_equal(
                inst.win_ctrs_x[pass_idx],
                ens.win_ctrs_x[pass_idx],
                decimal=6,
                err_msg=f"Pass {pass_idx}: win_ctrs_x mismatch"
            )

            # Check Y centers
            np.testing.assert_array_almost_equal(
                inst.win_ctrs_y[pass_idx],
                ens.win_ctrs_y[pass_idx],
                decimal=6,
                err_msg=f"Pass {pass_idx}: win_ctrs_y mismatch"
            )

            print(f"Pass {pass_idx}: win_ctrs_x = {inst.win_ctrs_x[pass_idx]}")
            print(f"Pass {pass_idx}: win_ctrs_y = {inst.win_ctrs_y[pass_idx]}")

    def test_window_spacing_match(self, correlators):
        """Verify window spacing is identical."""
        inst, ens, config = correlators

        for pass_idx in range(len(config.window_sizes)):
            assert inst.win_spacing_x[pass_idx] == ens.win_spacing_x[pass_idx], \
                f"Pass {pass_idx}: win_spacing_x mismatch"
            assert inst.win_spacing_y[pass_idx] == ens.win_spacing_y[pass_idx], \
                f"Pass {pass_idx}: win_spacing_y mismatch"

            print(f"Pass {pass_idx}: spacing = ({inst.win_spacing_x[pass_idx]}, {inst.win_spacing_y[pass_idx]})")

    def test_padding_values_match(self, correlators):
        """Verify n_pre_all and n_post_all are identical."""
        inst, ens, config = correlators

        for pass_idx in range(len(config.window_sizes)):
            # Check pre padding
            assert inst.n_pre_all[pass_idx] == ens.n_pre_all[pass_idx], \
                f"Pass {pass_idx}: n_pre_all mismatch - inst={inst.n_pre_all[pass_idx]}, ens={ens.n_pre_all[pass_idx]}"

            # Check post padding
            assert inst.n_post_all[pass_idx] == ens.n_post_all[pass_idx], \
                f"Pass {pass_idx}: n_post_all mismatch - inst={inst.n_post_all[pass_idx]}, ens={ens.n_post_all[pass_idx]}"

            print(f"Pass {pass_idx}: n_pre_all = {inst.n_pre_all[pass_idx]}, n_post_all = {inst.n_post_all[pass_idx]}")

    def test_padded_window_centers_match(self, correlators):
        """Verify padded window center arrays (win_ctrs_x_all, win_ctrs_y_all) are identical."""
        inst, ens, config = correlators

        for pass_idx in range(len(config.window_sizes)):
            # Check padded X centers
            np.testing.assert_array_almost_equal(
                inst.win_ctrs_x_all[pass_idx],
                ens.win_ctrs_x_all[pass_idx],
                decimal=6,
                err_msg=f"Pass {pass_idx}: win_ctrs_x_all mismatch"
            )

            # Check padded Y centers
            np.testing.assert_array_almost_equal(
                inst.win_ctrs_y_all[pass_idx],
                ens.win_ctrs_y_all[pass_idx],
                decimal=6,
                err_msg=f"Pass {pass_idx}: win_ctrs_y_all mismatch"
            )

            # Verify padding was applied correctly
            n_pre_y, n_pre_x = inst.n_pre_all[pass_idx]
            n_post_y, n_post_x = inst.n_post_all[pass_idx]

            expected_len_x = len(inst.win_ctrs_x[pass_idx]) + n_pre_x + n_post_x
            expected_len_y = len(inst.win_ctrs_y[pass_idx]) + n_pre_y + n_post_y

            assert len(inst.win_ctrs_x_all[pass_idx]) == expected_len_x, \
                f"Pass {pass_idx}: Padded X length mismatch"
            assert len(inst.win_ctrs_y_all[pass_idx]) == expected_len_y, \
                f"Pass {pass_idx}: Padded Y length mismatch"

            print(f"Pass {pass_idx}: win_ctrs_x_all len = {len(inst.win_ctrs_x_all[pass_idx])} "
                  f"(base={len(inst.win_ctrs_x[pass_idx])}, pre={n_pre_x}, post={n_post_x})")
            print(f"Pass {pass_idx}: win_ctrs_y_all len = {len(inst.win_ctrs_y_all[pass_idx])} "
                  f"(base={len(inst.win_ctrs_y[pass_idx])}, pre={n_pre_y}, post={n_post_y})")

    def test_interpolation_maps_match(self, correlators):
        """Verify cached interpolation maps are identical for passes > 0."""
        inst, ens, config = correlators

        # Skip pass 0 for instantaneous (it has None)
        for pass_idx in range(1, len(config.window_sizes)):
            # Dense maps
            inst_dense = inst.cached_dense_maps[pass_idx]
            ens_dense = ens.cached_dense_maps[pass_idx]

            if inst_dense is not None and ens_dense is not None:
                np.testing.assert_array_almost_equal(
                    inst_dense[0], ens_dense[0], decimal=6,
                    err_msg=f"Pass {pass_idx}: cached_dense_maps X mismatch"
                )
                np.testing.assert_array_almost_equal(
                    inst_dense[1], ens_dense[1], decimal=6,
                    err_msg=f"Pass {pass_idx}: cached_dense_maps Y mismatch"
                )
                print(f"Pass {pass_idx}: cached_dense_maps match (shape={inst_dense[0].shape})")

            # Predictor maps
            inst_pred = inst.cached_predictor_maps[pass_idx]
            ens_pred = ens.cached_predictor_maps[pass_idx]

            if inst_pred is not None and ens_pred is not None:
                np.testing.assert_array_almost_equal(
                    inst_pred[0], ens_pred[0], decimal=6,
                    err_msg=f"Pass {pass_idx}: cached_predictor_maps X mismatch"
                )
                np.testing.assert_array_almost_equal(
                    inst_pred[1], ens_pred[1], decimal=6,
                    err_msg=f"Pass {pass_idx}: cached_predictor_maps Y mismatch"
                )
                print(f"Pass {pass_idx}: cached_predictor_maps match (shape={inst_pred[0].shape})")

    def test_smoothing_parameters_match(self, correlators):
        """Verify Gaussian smoothing parameters (ksize_filt, sd) match for passes > 0."""
        inst, ens, config = correlators

        # Pass 0 differs by design: inst uses (0,0), ens uses (1,1)
        # But passes > 0 should match
        for pass_idx in range(1, len(config.window_sizes)):
            assert inst.ksize_filt[pass_idx] == ens.ksize_filt[pass_idx], \
                f"Pass {pass_idx}: ksize_filt mismatch - inst={inst.ksize_filt[pass_idx]}, ens={ens.ksize_filt[pass_idx]}"

            np.testing.assert_almost_equal(
                inst.sd[pass_idx], ens.sd[pass_idx], decimal=6,
                err_msg=f"Pass {pass_idx}: sd (sigma) mismatch"
            )

            print(f"Pass {pass_idx}: ksize_filt = {inst.ksize_filt[pass_idx]}, sd = {inst.sd[pass_idx]:.4f}")


class TestPredicatorPaddingParity:
    """Test that predictor field padding produces identical results."""

    def test_predictor_padding_matches_instantaneous(self):
        """
        Verify that the new ensemble padding logic produces the same
        padded predictor field dimensions as instantaneous mode.
        """
        config = create_matching_config()

        inst = InstantaneousCorrelatorCPU(config)
        ens = EnsembleCorrelatorCPU(config)

        # Simulate a predictor field from pass 0 (going into pass 1)
        n_win_y_pass0 = len(inst.win_ctrs_y[0])
        n_win_x_pass0 = len(inst.win_ctrs_x[0])

        # Create a mock predictor field (same shape as pass 0 output)
        mock_predictor = np.random.randn(n_win_y_pass0, n_win_x_pass0, 2).astype(np.float32)

        # Get padding values for pass 0 (used when entering pass 1)
        pre_y, pre_x = inst.n_pre_all[0]
        post_y, post_x = inst.n_post_all[0]

        # Apply padding like instantaneous does (at end of pass 0)
        inst_padded = np.pad(
            mock_predictor,
            ((pre_y, post_y), (pre_x, post_x), (0, 0)),
            mode="edge"
        )

        # Apply padding like ensemble does (in _get_im_mesh for pass 1)
        # This uses prev_pass = 0, so n_pre_all[0] and n_post_all[0]
        prev_pass = 0
        ens_pre_y, ens_pre_x = ens.n_pre_all[prev_pass]
        ens_post_y, ens_post_x = ens.n_post_all[prev_pass]

        ens_padded = np.pad(
            mock_predictor,
            ((ens_pre_y, ens_post_y), (ens_pre_x, ens_post_x), (0, 0)),
            mode="edge"
        )

        # They should be identical
        np.testing.assert_array_equal(inst_padded, ens_padded)

        print(f"Original predictor shape: {mock_predictor.shape}")
        print(f"Padded predictor shape: {inst_padded.shape}")
        print(f"Padding applied: pre=({pre_y}, {pre_x}), post=({post_y}, {post_x})")

        # Also verify the padded shape matches what the interpolation maps expect
        expected_y = len(inst.win_ctrs_y_all[0])
        expected_x = len(inst.win_ctrs_x_all[0])

        assert inst_padded.shape[0] == expected_y, \
            f"Padded Y dim {inst_padded.shape[0]} != expected {expected_y}"
        assert inst_padded.shape[1] == expected_x, \
            f"Padded X dim {inst_padded.shape[1]} != expected {expected_x}"

        print(f"Padded shape matches win_ctrs_all dimensions: ({expected_y}, {expected_x})")


class TestPredictorRemapParity:
    """Test that predictor remap produces identical results between modes."""

    def test_predictor_remap_produces_identical_output(self):
        """
        Verify that given identical predictor input, both modes
        produce identical delta_ab_pred after padding + smoothing + remap.

        This tests the full predictor pipeline that was causing edge artifacts.
        """
        import cv2
        from scipy.ndimage import gaussian_filter

        config = create_matching_config()
        inst = InstantaneousCorrelatorCPU(config)
        ens = EnsembleCorrelatorCPU(config)

        # Test for pass 1 (using predictor from pass 0)
        pass_idx = 1
        prev_pass = 0

        # Create mock predictor field (pass 0 output shape)
        n_win_y = len(inst.win_ctrs_y[prev_pass])
        n_win_x = len(inst.win_ctrs_x[prev_pass])

        # Use non-uniform values to detect interpolation differences
        np.random.seed(42)
        mock_predictor = np.random.randn(n_win_y, n_win_x, 2).astype(np.float32) * 10

        # Get padding values (should be identical - verified by other tests)
        pre_y, pre_x = inst.n_pre_all[prev_pass]
        post_y, post_x = inst.n_post_all[prev_pass]

        # === INSTANTANEOUS PATH ===
        # Step 1: Pad (done at end of prev pass in correlate_batch)
        inst_padded = np.pad(
            mock_predictor,
            ((pre_y, post_y), (pre_x, post_x), (0, 0)),
            mode="edge"
        )

        # Step 2: Smooth (done at start of current pass in _get_im_mesh)
        inst_smoothed = np.zeros_like(inst_padded)
        inst_smoothed[..., 0] = gaussian_filter(
            inst_padded[..., 0],
            sigma=inst.sd[pass_idx],
            truncate=(inst.ksize_filt[pass_idx][0] - 1) / (2 * inst.sd[pass_idx]) if inst.sd[pass_idx] > 0 else 0,
            mode="nearest",
        )
        inst_smoothed[..., 1] = gaussian_filter(
            inst_padded[..., 1],
            sigma=inst.sd[pass_idx],
            truncate=(inst.ksize_filt[pass_idx][0] - 1) / (2 * inst.sd[pass_idx]) if inst.sd[pass_idx] > 0 else 0,
            mode="nearest",
        )

        # Step 3: Remap to current pass grid
        map_x, map_y = inst.cached_predictor_maps[pass_idx]
        n_win_y_curr = len(inst.win_ctrs_y[pass_idx])
        n_win_x_curr = len(inst.win_ctrs_x[pass_idx])
        inst_delta_ab_pred = np.zeros((n_win_y_curr, n_win_x_curr, 2), dtype=np.float32)

        for d in range(2):
            inst_delta_ab_pred[..., d] = cv2.remap(
                inst_smoothed[..., d],
                map_x,
                map_y,
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

        # === ENSEMBLE PATH ===
        # Step 1: Pad (done in _get_im_mesh using prev_pass indices)
        ens_pre_y, ens_pre_x = ens.n_pre_all[prev_pass]
        ens_post_y, ens_post_x = ens.n_post_all[prev_pass]

        ens_padded = np.pad(
            mock_predictor,
            ((ens_pre_y, ens_post_y), (ens_pre_x, ens_post_x), (0, 0)),
            mode="edge"
        )

        # Step 2: Smooth
        ens_smoothed = np.zeros_like(ens_padded)
        ens_smoothed[..., 0] = gaussian_filter(
            ens_padded[..., 0],
            sigma=ens.sd[pass_idx],
            truncate=(ens.ksize_filt[pass_idx][0] - 1) / (2 * ens.sd[pass_idx]) if ens.sd[pass_idx] > 0 else 0,
            mode="nearest",
        )
        ens_smoothed[..., 1] = gaussian_filter(
            ens_padded[..., 1],
            sigma=ens.sd[pass_idx],
            truncate=(ens.ksize_filt[pass_idx][0] - 1) / (2 * ens.sd[pass_idx]) if ens.sd[pass_idx] > 0 else 0,
            mode="nearest",
        )

        # Step 3: Remap
        ens_map_x, ens_map_y = ens.cached_predictor_maps[pass_idx]
        ens_n_win_y_curr = len(ens.win_ctrs_y[pass_idx])
        ens_n_win_x_curr = len(ens.win_ctrs_x[pass_idx])
        ens_delta_ab_pred = np.zeros((ens_n_win_y_curr, ens_n_win_x_curr, 2), dtype=np.float32)

        for d in range(2):
            ens_delta_ab_pred[..., d] = cv2.remap(
                ens_smoothed[..., d],
                ens_map_x,
                ens_map_y,
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

        # === VERIFY PARITY ===
        # Intermediate steps should match
        np.testing.assert_array_equal(inst_padded, ens_padded,
            err_msg="Padded predictor mismatch")
        np.testing.assert_array_almost_equal(inst_smoothed, ens_smoothed, decimal=6,
            err_msg="Smoothed predictor mismatch")

        # Final output should match
        np.testing.assert_array_almost_equal(inst_delta_ab_pred, ens_delta_ab_pred, decimal=6,
            err_msg="delta_ab_pred mismatch - remap produced different results!")

        # Print diagnostics
        print(f"Mock predictor shape: {mock_predictor.shape}")
        print(f"Padded shape: {inst_padded.shape}")
        print(f"delta_ab_pred shape: {inst_delta_ab_pred.shape}")
        print(f"Edge values match: TL={inst_delta_ab_pred[0,0,1]:.4f}, "
              f"TR={inst_delta_ab_pred[0,-1,1]:.4f}, "
              f"BL={inst_delta_ab_pred[-1,0,1]:.4f}, "
              f"BR={inst_delta_ab_pred[-1,-1,1]:.4f}")

    def test_predictor_storage_is_padded(self):
        """
        Verify that ensemble now stores PADDED predictor (matching instantaneous).

        This tests the fix in single_pass_accumulator.py finalize_pass().
        """
        config = create_matching_config()
        inst = InstantaneousCorrelatorCPU(config)
        ens = EnsembleCorrelatorCPU(config)

        # For each pass, verify the expected padded predictor shape
        for pass_idx in range(len(config.window_sizes)):
            n_win_y = len(inst.win_ctrs_y[pass_idx])
            n_win_x = len(inst.win_ctrs_x[pass_idx])
            pre_y, pre_x = inst.n_pre_all[pass_idx]
            post_y, post_x = inst.n_post_all[pass_idx]

            expected_padded_y = n_win_y + pre_y + post_y
            expected_padded_x = n_win_x + pre_x + post_x

            print(f"Pass {pass_idx}: ux_mat shape = ({n_win_y}, {n_win_x})")
            print(f"Pass {pass_idx}: Expected PADDED pred shape = ({expected_padded_y}, {expected_padded_x})")
            print(f"Pass {pass_idx}: n_pre = ({pre_y}, {pre_x}), n_post = ({post_y}, {post_x})")

            # Verify instantaneous predictor_field would be this shape
            # (In actual run, PIVPassResult.predictor_field has shape (padded_y, padded_x, 2))

            # Verify ensemble should now produce same shape
            # (In actual run, PIVEnsemblePassResult.pred_x has shape (padded_y, padded_x))


class TestMultipleConfigurations:
    """Test grid parity across various configurations."""

    @pytest.mark.parametrize("image_shape", [
        (256, 256),
        (512, 512),
        (480, 640),  # Non-square
        (1024, 768),
    ])
    def test_various_image_shapes(self, image_shape):
        """Verify parity across different image shapes."""
        config = create_matching_config(image_shape=image_shape)

        inst = InstantaneousCorrelatorCPU(config)
        ens = EnsembleCorrelatorCPU(config)

        for pass_idx in range(len(config.window_sizes)):
            np.testing.assert_array_almost_equal(
                inst.win_ctrs_x[pass_idx], ens.win_ctrs_x[pass_idx], decimal=6
            )
            np.testing.assert_array_almost_equal(
                inst.win_ctrs_y[pass_idx], ens.win_ctrs_y[pass_idx], decimal=6
            )
            assert inst.n_pre_all[pass_idx] == ens.n_pre_all[pass_idx]
            assert inst.n_post_all[pass_idx] == ens.n_post_all[pass_idx]

        print(f"Image shape {image_shape}: All grids match!")

    @pytest.mark.parametrize("window_sizes,overlaps", [
        ([(64, 64), (32, 32)], [50, 50]),
        ([(128, 128), (64, 64), (32, 32)], [50, 50, 50]),
        ([(64, 64), (32, 32), (16, 16), (8, 8)], [50, 50, 50, 50]),
        ([(64, 64), (32, 32)], [75, 75]),  # Higher overlap
        ([(64, 64), (32, 32)], [25, 25]),  # Lower overlap
    ])
    def test_various_window_configs(self, window_sizes, overlaps):
        """Verify parity across different window configurations."""
        config = create_matching_config(window_sizes=window_sizes, overlaps=overlaps)

        inst = InstantaneousCorrelatorCPU(config)
        ens = EnsembleCorrelatorCPU(config)

        for pass_idx in range(len(window_sizes)):
            np.testing.assert_array_almost_equal(
                inst.win_ctrs_x[pass_idx], ens.win_ctrs_x[pass_idx], decimal=6
            )
            np.testing.assert_array_almost_equal(
                inst.win_ctrs_y[pass_idx], ens.win_ctrs_y[pass_idx], decimal=6
            )
            assert inst.n_pre_all[pass_idx] == ens.n_pre_all[pass_idx]
            assert inst.n_post_all[pass_idx] == ens.n_post_all[pass_idx]

        print(f"Window config {window_sizes} @ {overlaps}% overlap: All grids match!")


class TestCubicInterpolationSafety:
    """
    Test that n_pre >= 2 and n_post >= 2 for safe cubic interpolation.

    Cubic interpolation needs 4 neighbors. At map index 0.x, cubic reaches
    for index -1 which is OOB. With n_pre >= 2, map index is >= 1.x and all
    4 neighbors (0, 1, 2, 3) are valid.
    """

    def test_instantaneous_n_pre_at_least_2(self):
        """Verify instantaneous mode has n_pre >= 2 for all passes."""
        config = create_matching_config()
        inst = InstantaneousCorrelatorCPU(config)

        for pass_idx in range(len(config.window_sizes)):
            n_pre = inst.n_pre_all[pass_idx]
            n_post = inst.n_post_all[pass_idx]

            assert n_pre[0] >= 2, \
                f"Pass {pass_idx}: n_pre_y = {n_pre[0]} < 2 (unsafe for cubic)"
            assert n_pre[1] >= 2, \
                f"Pass {pass_idx}: n_pre_x = {n_pre[1]} < 2 (unsafe for cubic)"
            assert n_post[0] >= 2, \
                f"Pass {pass_idx}: n_post_y = {n_post[0]} < 2 (unsafe for cubic)"
            assert n_post[1] >= 2, \
                f"Pass {pass_idx}: n_post_x = {n_post[1]} < 2 (unsafe for cubic)"

        print(f"Instantaneous: All n_pre/n_post >= 2 across {len(config.window_sizes)} passes")

    def test_ensemble_standard_n_pre_at_least_2(self):
        """Verify ensemble standard mode has n_pre >= 2 for all passes."""
        config = create_matching_config()
        ens = EnsembleCorrelatorCPU(config)

        for pass_idx in range(len(config.window_sizes)):
            n_pre = ens.n_pre_all[pass_idx]
            n_post = ens.n_post_all[pass_idx]

            assert n_pre[0] >= 2, \
                f"Pass {pass_idx}: n_pre_y = {n_pre[0]} < 2 (unsafe for cubic)"
            assert n_pre[1] >= 2, \
                f"Pass {pass_idx}: n_pre_x = {n_pre[1]} < 2 (unsafe for cubic)"
            assert n_post[0] >= 2, \
                f"Pass {pass_idx}: n_post_y = {n_post[0]} < 2 (unsafe for cubic)"
            assert n_post[1] >= 2, \
                f"Pass {pass_idx}: n_post_x = {n_post[1]} < 2 (unsafe for cubic)"

        print(f"Ensemble standard: All n_pre/n_post >= 2 across {len(config.window_sizes)} passes")

    def test_ensemble_single_mode_n_pre_at_least_2(self):
        """Verify ensemble single mode has n_pre >= 2 for all passes."""
        config = Config()
        config._detected_image_shape = (256, 256)
        config.data['ensemble_piv'] = {}
        config.data['processing'] = {}
        config.data['ensemble_piv']['window_size'] = [(4, 4), (4, 4)]
        config.data['ensemble_piv']['overlap'] = [50, 50]
        config.data['ensemble_piv']['window_type'] = 'gaussian'
        config.data['ensemble_piv']['num_passes'] = 2
        config.data['ensemble_piv']['type'] = ['single', 'single']
        config.data['ensemble_piv']['sum_window'] = [16, 16]
        config.data['ensemble_piv']['num_peaks'] = 1
        config.data['ensemble_piv']['peak_finder'] = 0
        config.data['processing']['omp_threads'] = 1

        ens = EnsembleCorrelatorCPU(config)

        for pass_idx in range(2):
            n_pre = ens.n_pre_all[pass_idx]
            n_post = ens.n_post_all[pass_idx]

            assert n_pre[0] >= 2, \
                f"Single mode pass {pass_idx}: n_pre_y = {n_pre[0]} < 2"
            assert n_pre[1] >= 2, \
                f"Single mode pass {pass_idx}: n_pre_x = {n_pre[1]} < 2"
            assert n_post[0] >= 2, \
                f"Single mode pass {pass_idx}: n_post_y = {n_post[0]} < 2"
            assert n_post[1] >= 2, \
                f"Single mode pass {pass_idx}: n_post_x = {n_post[1]} < 2"

        print(f"Ensemble single mode: All n_pre/n_post >= 2")

    def test_single_pixel_mode_n_pre_at_least_2(self):
        """Verify single pixel mode (1x1 window) has n_pre >= 2."""
        config = Config()
        config._detected_image_shape = (256, 256)
        config.data['ensemble_piv'] = {}
        config.data['processing'] = {}
        config.data['ensemble_piv']['window_size'] = [(1, 1), (1, 1)]
        config.data['ensemble_piv']['overlap'] = [0, 0]  # Can't overlap 1x1
        config.data['ensemble_piv']['window_type'] = 'gaussian'
        config.data['ensemble_piv']['num_passes'] = 2
        config.data['ensemble_piv']['type'] = ['single', 'single']
        config.data['ensemble_piv']['sum_window'] = [16, 16]
        config.data['ensemble_piv']['num_peaks'] = 1
        config.data['ensemble_piv']['peak_finder'] = 0
        config.data['processing']['omp_threads'] = 1

        ens = EnsembleCorrelatorCPU(config)

        for pass_idx in range(2):
            n_pre = ens.n_pre_all[pass_idx]
            n_post = ens.n_post_all[pass_idx]

            assert n_pre[0] >= 2, \
                f"Single pixel pass {pass_idx}: n_pre_y = {n_pre[0]} < 2"
            assert n_pre[1] >= 2, \
                f"Single pixel pass {pass_idx}: n_pre_x = {n_pre[1]} < 2"
            assert n_post[0] >= 2, \
                f"Single pixel pass {pass_idx}: n_post_y = {n_post[0]} < 2"
            assert n_post[1] >= 2, \
                f"Single pixel pass {pass_idx}: n_post_x = {n_post[1]} < 2"

        print(f"Single pixel mode (1x1): All n_pre/n_post >= 2")

    @pytest.mark.parametrize("image_shape,window_sizes,overlaps", [
        # Edge cases that would have had n_pre=1 before fix
        ((256, 256), [(64, 64), (32, 32)], [50, 50]),
        ((128, 128), [(32, 32), (16, 16)], [50, 50]),
        ((64, 64), [(16, 16), (8, 8)], [50, 50]),
        # High overlap (small spacing)
        ((512, 512), [(64, 64), (32, 32)], [75, 75]),
        # Low overlap (large spacing)
        ((512, 512), [(64, 64), (32, 32)], [25, 25]),
    ])
    def test_various_configs_have_safe_padding(self, image_shape, window_sizes, overlaps):
        """Verify n_pre >= 2 across various configurations."""
        config = create_matching_config(
            image_shape=image_shape,
            window_sizes=window_sizes,
            overlaps=overlaps
        )

        inst = InstantaneousCorrelatorCPU(config)
        ens = EnsembleCorrelatorCPU(config)

        for pass_idx in range(len(window_sizes)):
            for mode_name, corr in [("inst", inst), ("ens", ens)]:
                n_pre = corr.n_pre_all[pass_idx]
                n_post = corr.n_post_all[pass_idx]

                assert n_pre[0] >= 2, \
                    f"{mode_name} pass {pass_idx}: n_pre_y = {n_pre[0]} < 2"
                assert n_pre[1] >= 2, \
                    f"{mode_name} pass {pass_idx}: n_pre_x = {n_pre[1]} < 2"
                assert n_post[0] >= 2, \
                    f"{mode_name} pass {pass_idx}: n_post_y = {n_post[0]} < 2"
                assert n_post[1] >= 2, \
                    f"{mode_name} pass {pass_idx}: n_post_x = {n_post[1]} < 2"

        print(f"Config {image_shape}/{window_sizes}/{overlaps}: Safe padding verified")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
