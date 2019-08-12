import unittest
import numpy as np
import tensorflow as tf

from tf_ops.nn_distance import tf_nndistance
from monopsr.core import distance_metrics


class NearestNeighborTest(unittest.TestCase):

    def test_nn_distance(self):
        """Test for nearest neighbor algorithm where distance should be 0.
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]]
        point_cloud_2 = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(tf_point_cloud_1, tf_point_cloud_2)

        with tf.Session() as sess:
            dist1, idx1 = sess.run([dist1, idx1])

        np.testing.assert_almost_equal(np.sum(dist1), 0)
        np.testing.assert_equal(idx1, [[0, 1, 2]])

    def test_nn_distance_2(self):
        """Test for nearest neighbor algorithm where distance is non-zero.
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]]
        point_cloud_2 = [[[1., 1., 1.], [2., 2., 2.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(tf_point_cloud_1, tf_point_cloud_2)

        with tf.Session() as sess:
            dist1, idx1 = sess.run([dist1, idx1])

        np.testing.assert_almost_equal(np.sum(dist1), 3.0)
        np.testing.assert_equal(idx1, [[0, 1, 1]])

    def test_nn_distance_negative(self):
        """Test negative point cloud values for computing nearest neighbor squared distance.
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[-2., 2., -2.], [1., 3., 4.]]]
        point_cloud_2 = [[[2., 0., 2.], [3., -5., 7.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(tf_point_cloud_1, tf_point_cloud_2)

        with tf.Session() as sess:
            dist1, idx1 = sess.run([dist1, idx1])

        np.testing.assert_almost_equal(np.sum(dist1), 50.0)

    def test_nn_distance_batch(self):
        """Tests batches for computing nearest neighbor squared distance.
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]],
                         [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]]
        point_cloud_2 = [[[1., 0., 1.], [2., 0., 2.], [3., 0., 3.]],
                         [[4., 4., 4.], [2., 2., 2.], [3., 3., 3.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(tf_point_cloud_1, tf_point_cloud_2)

        with tf.Session() as sess:
            dist1, idx1 = sess.run([dist1, idx1])

        np.testing.assert_almost_equal(np.sum(dist1, axis=1), [14.0, 3.0])

    def test_sklearn_vs_tf_nn_calc(self):
        """Test to see if our implementation of chamfer distance produces same result as sklearn
        """

        # Create test point clouds of shape [batch_size, n_points, 3]
        point_cloud_1 = [[[1., 1., 1.], [2., 2., 2.], [1., 5., 7.]]]
        point_cloud_2 = [[[1., 5., 7.], [10., 0., 5.]]]

        tf_point_cloud_1 = tf.constant(point_cloud_1)
        tf_point_cloud_2 = tf.constant(point_cloud_2)

        dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(tf_point_cloud_1, tf_point_cloud_2)

        with tf.Session() as sess:
            dist1, idx1, dist2, idx2 = sess.run([dist1, idx1, dist2, idx2])
            chamfer_dist_calc1 = np.sum(dist1) + np.sum(dist2)

        chamfer_dist_calc2 = distance_metrics.calc_chamfer_dist(point_cloud_1[0], point_cloud_2[0])

        np.testing.assert_approx_equal(chamfer_dist_calc1, chamfer_dist_calc2)
