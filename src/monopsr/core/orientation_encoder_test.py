import numpy as np
import tensorflow as tf

from monopsr.core import orientation_encoder


class OrientationEncoderTest(tf.test.TestCase):

    def test_np_wrap_to_pi(self):
        """Test wrapping angles to [-pi, pi]"""
        angles_deg = np.asarray([-360, -185, -175, -90, 0, 90, 175, 185, 360])
        angles_rad = np.deg2rad(angles_deg)

        exp_angles_deg = np.asarray([0, 175, -175, -90, 0, 90, 175, -175, 0])
        exp_angles_rad = np.deg2rad(exp_angles_deg)

        angles_wrapped = orientation_encoder.np_wrap_to_pi(angles_rad)

        np.testing.assert_equal(angles_wrapped, exp_angles_rad)

    def test_np_orientation_to_angle_bin_centers(self):
        """Test 8 angle bins and residuals
        """
        num_bins = 8

        angle_bin_centers = [
            np.deg2rad(-180.0), np.deg2rad(-135.0),
            np.deg2rad(-90.0), np.deg2rad(-45.0),
            np.deg2rad(0.0),
            np.deg2rad(45.0), np.deg2rad(90.0),
            np.deg2rad(135.0), np.deg2rad(180.0),
        ]

        angle_bins, residuals, one_hot_valid_bins = \
            zip(*[orientation_encoder.np_orientation_to_angle_bin(orientation, num_bins)
                  for orientation in angle_bin_centers])

        residuals = np.asarray([np.take(residual, np.argmax(one_hot))
                                for residual, one_hot in zip(residuals, one_hot_valid_bins)])

        exp_angle_bins = [4, 5, 6, 7, 0, 1, 2, 3, 4]
        np.testing.assert_equal(angle_bins, exp_angle_bins)
        np.testing.assert_equal(residuals, np.zeros_like(angle_bin_centers))

    def test_test_np_orientation_to_angle_bin_residuals(self):

        num_bins = 8
        orientations = [
            np.deg2rad(-181), np.deg2rad(-179),
            np.deg2rad(-136), np.deg2rad(-134),
            np.deg2rad(-91), np.deg2rad(-89),
            np.deg2rad(-46), np.deg2rad(-44),
            np.deg2rad(-1), np.deg2rad(1),
            np.deg2rad(44), np.deg2rad(46),
            np.deg2rad(89), np.deg2rad(91),
            np.deg2rad(134), np.deg2rad(136),
            np.deg2rad(179), np.deg2rad(181),
        ]

        angle_bins, residuals, one_hot_valid_bins = \
            zip(*[orientation_encoder.np_orientation_to_angle_bin(orientation, num_bins)
                  for orientation in orientations])

        residuals = np.asarray([np.take(residual, np.argmax(one_hot))
                                for residual, one_hot in zip(residuals, one_hot_valid_bins)])

        exp_angle_bins = [
            4, 4,
            5, 5,
            6, 6,
            7, 7,
            0, 0,
            1, 1,
            2, 2,
            3, 3,
            4, 4,
        ]
        exp_residuals = [np.deg2rad(-1.0), np.deg2rad(1.0)] * 9

        np.testing.assert_equal(angle_bins, exp_angle_bins)
        np.testing.assert_almost_equal(residuals, exp_residuals)

    def test_np_angle_bin_to_orientation_random(self):
        num_bins = 8
        angle_bins_to_test = [
            (0, 0.0),
            (0, np.deg2rad(10.0)),
            (4, np.deg2rad(-10.0)),
            (7, np.deg2rad(50.0)),
        ]

        exp_orientations = [
            0.0,
            np.deg2rad(10.0),
            np.deg2rad(170.0),
            np.deg2rad(5.0),
        ]

        for (angle_bin, residual), (exp_orientation) in zip(angle_bins_to_test, exp_orientations):
            orientation = orientation_encoder.np_angle_bin_to_orientation(
                angle_bin, residual, num_bins)

            np.testing.assert_approx_equal(orientation, exp_orientation)

    def test_tf_orientation_to_angle_vector(self):
        # Test conversion for angles between [-pi, pi] with 0.5 degree steps
        np_orientations = np.arange(-np.pi, np.pi, np.pi / 360.0)

        expected_angle_vectors = np.stack([np.cos(np_orientations),
                                           np.sin(np_orientations)], axis=1)

        # Convert to tensors and convert to angle unit vectors
        tf_orientations = tf.convert_to_tensor(np_orientations)
        tf_angle_vectors = orientation_encoder.tf_orientation_to_angle_vector(
            tf_orientations)

        with self.test_session() as sess:
            angle_vectors_out = sess.run(tf_angle_vectors)

            np.testing.assert_allclose(angle_vectors_out,
                                       expected_angle_vectors)

    def test_angle_vectors_to_orientation(self):
        # Test conversion for angles between [-pi, pi] with 0.5 degree steps
        np_angle_vectors = \
            np.asarray([[np.cos(angle), np.sin(angle)]
                        for angle in np.arange(-np.pi, np.pi, np.pi / 360.0)])

        # Check that tf output matches numpy's arctan2 output
        expected_orientations = np.arctan2(np_angle_vectors[:, 1],
                                           np_angle_vectors[:, 0])

        # Convert to tensors and convert to orientation angles
        tf_angle_vectors = tf.convert_to_tensor(np_angle_vectors)
        tf_orientations = orientation_encoder.tf_angle_vector_to_orientation(
            tf_angle_vectors)

        with self.test_session() as sess:
            orientations_out = sess.run(tf_orientations)
            np.testing.assert_allclose(orientations_out,
                                       expected_orientations)

    def test_zeros_angle_vectors_to_orientation(self):
        # Test conversion for angle vectors with zeros in them
        np_angle_vectors = np.asarray(
            [[0, 0],
             [1, 0], [10, 0],
             [0, 1], [0, 10],
             [-1, 0], [-10, 0],
             [0, -1], [0, -10]])

        half_pi = np.pi / 2
        expected_orientations = [0,
                                 0, 0,
                                 half_pi, half_pi,
                                 np.pi, np.pi,
                                 -half_pi, -half_pi]

        # Convert to tensors and convert to orientation angles
        tf_angle_vectors = tf.convert_to_tensor(np_angle_vectors,
                                                dtype=tf.float64)
        tf_orientations = orientation_encoder.tf_angle_vector_to_orientation(
            tf_angle_vectors)

        with self.test_session() as sess:
            orientations_out = sess.run(tf_orientations)
            np.testing.assert_allclose(orientations_out,
                                       expected_orientations)

    def test_two_way_conversion(self):
        # Test conversion for angles between [-pi, pi] with 0.5 degree steps
        np_orientations = np.arange(np.pi, np.pi, np.pi / 360.0)

        tf_angle_vectors = orientation_encoder.tf_orientation_to_angle_vector(
            np_orientations)
        tf_orientations = orientation_encoder.tf_angle_vector_to_orientation(
            tf_angle_vectors)

        # Check that conversion from orientation -> angle vector ->
        # orientation results in the same values
        with self.test_session() as sess:
            orientations_out = sess.run(tf_orientations)
            np.testing.assert_allclose(orientations_out,
                                       np_orientations)

    def test_np_overlap_bins(self):
        # Test a general case for overlapping bins when the first and second bins overlap.

        num_bins = 4
        orientation = np.deg2rad(43)
        overlap = np.deg2rad(10)

        angle_bin, residual, valid_bins = orientation_encoder.np_orientation_to_angle_bin(
            orientation, num_bins, overlap)

        gt_valid_bins = [1, 1, 0, 0]

        np.testing.assert_allclose(valid_bins, gt_valid_bins)
        np.testing.assert_approx_equal(angle_bin, 0)

    def test_np_overlap_bins_lower_edge(self):
        # Test overlapping bins when the angle is in the last bin and overlaps with the first bin

        num_bins = 4
        orientation = np.deg2rad(-43)
        overlap = np.deg2rad(10)

        angle_bin, residual, valid_bins = orientation_encoder.np_orientation_to_angle_bin(
            orientation, num_bins, overlap)

        gt_valid_bins = [1, 0, 0, 1]

        np.testing.assert_allclose(valid_bins, gt_valid_bins)

    def test_np_overlap_bins_upper_edge(self):
        # Test overlapping bins when the angle is in the first bin and overlaps with the last bin

        num_bins = 4
        orientation = np.deg2rad(310)
        overlap = np.deg2rad(10)

        angle_bin, residual, valid_bins = orientation_encoder.np_orientation_to_angle_bin(
            orientation, num_bins, overlap)

        gt_valid_bins = [1, 0, 0, 1]

        np.testing.assert_allclose(valid_bins, gt_valid_bins)

    def test_np_overlap_bins_multiple_residuals(self):
        # Test the calculation of residuals for all bins

        num_bins = 4
        orientation = np.deg2rad(0)
        overlap = np.deg2rad(10)

        angle_bin, residual, valid_bins = orientation_encoder.np_orientation_to_angle_bin(
            orientation, num_bins, overlap)

        gt_valid_bins = [1, 0, 0, 0]
        gt_residual = [0, -np.deg2rad(90), -np.deg2rad(180), np.deg2rad(90)]

        np.testing.assert_allclose(valid_bins, gt_valid_bins)
        np.testing.assert_allclose(residual, gt_residual)
