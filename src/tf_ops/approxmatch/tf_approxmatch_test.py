import unittest
import numpy as np
import tensorflow as tf

from tf_ops.approxmatch import tf_approxmatch


class ApproxMatchTest(unittest.TestCase):

    def test_emd(self):
        """Test for the approximate algorithm for computing the distance where loss
        should be zero.
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]]
        point_cloud_2 = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        match = tf_approxmatch.approx_match(tf_point_cloud_1, tf_point_cloud_2)
        distance = tf.reduce_mean(tf_approxmatch.match_cost(point_cloud_1, point_cloud_2, match))

        with tf.Session() as sess:
            distance = sess.run([distance])

        np.testing.assert_almost_equal(distance, 0)

    def test_emd_2(self):
        """Test for the approximate algorithm for computing the Earth Mover's Distance to see
        if match selects closest point, and if the loss is reasonable.
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]]
        point_cloud_2 = [[[1., 0., 1.], [2., 0., 2.], [3., 0., 3.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        match = tf_approxmatch.approx_match(tf_point_cloud_1, tf_point_cloud_2)
        distance = tf.reduce_mean(tf_approxmatch.match_cost(point_cloud_1, point_cloud_2, match))

        with tf.Session() as sess:
            match, distance = sess.run([match, distance])

        matched_indices = np.argmax(np.squeeze(match), axis=1)
        np.testing.assert_equal(matched_indices, [0, 1, 2])
        np.testing.assert_almost_equal(distance, 6.0, decimal=2)

    def test_emd_negative(self):
        """Test negative point cloud values for computing the approximate Earth Mover's Distance.
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[-2., 2., -2.]]]
        point_cloud_2 = [[[2., 0., 2.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        match = tf_approxmatch.approx_match(tf_point_cloud_1, tf_point_cloud_2)
        distance = tf.reduce_mean(tf_approxmatch.match_cost(point_cloud_1, point_cloud_2, match))

        with tf.Session() as sess:
            match, distance = sess.run([match, distance])

        np.testing.assert_almost_equal(distance, 6.0, decimal=2)

    def test_emd_batch(self):
        """Tests batches for the approximate algorithm for computing the Earth Mover's Distance.
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]],
                         [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]]
        point_cloud_2 = [[[1., 0., 1.], [2., 0., 2.], [3., 0., 3.]],
                         [[4., 4., 4.], [2., 2., 2.], [3., 3., 3.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        match = tf_approxmatch.approx_match(tf_point_cloud_1, tf_point_cloud_2)
        distance = tf_approxmatch.match_cost(point_cloud_1, point_cloud_2, match)

        with tf.Session() as sess:
            match, distance = sess.run([match, distance])

        np.testing.assert_almost_equal(distance, [6.0, 5.196152], decimal=2)
